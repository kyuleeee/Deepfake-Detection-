import os
import argparse
import torch
from models.pred_func import load_genconvit, pred_vid, df_face, face_rec, preprocess_frame, is_video, save_detected_faces_png_named
from models.config import load_config
from typing import List, Optional, Dict, Union
import csv
import numpy as np
from PIL import Image
from tqdm import tqdm

config = load_config()


def save_results_to_csv(
    results: List[Dict[str, Union[str, int, List[float]]]],
    filepath: str,
    for_submit=False
) -> None:
    # values and keys are guaranteed; use the fastest path (csv.writer + direct indexing)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        row = w.writerow  # micro-optimization: cache attribute lookup
        row(("filename", "label") if for_submit else (
            "name", "pred", "pred_proba"))
        if for_submit:
            for r in results:
                row((r['name'], r['pred']))
        else:
            for r in results:
                row((r["name"], r["pred"], r["pred_proba"]))


def get_files_paths(main_path: str, exs: List[str]) -> Dict[str, List[str]]:
    exs = {("." + ext.lstrip(".")).lower() for ext in exs}
    results = []

    for root, _, files in os.walk(main_path):
        for fname in files:
            abs_path = os.path.abspath(os.path.join(root, fname))
            _, ext = os.path.splitext(fname)
            if ext.lower() in exs:
                results.append(abs_path)
            else:
                print(f"{fname} is not in [{exs}]. passing...")

    return results


def predict_video(
    file_path: str,
    model,
    fp16_mode: bool = False,
    num_frames: int = 15,
):
    """
    1. num_frames 마다 프레임을 균등하게 추출하고
    2. 얼굴을 검출하고 224x224로 크롭한 다음
    3. N개의 프레임을 가지는 크기 (N, 3, 224, 224)의 정규화된 텐서를 반환.
    만약 얼굴이 검출되지 않았다면 크기가 0인 텐서를 반환한다.
    """
    df = df_face(file_path, num_frames)

    if fp16_mode and df.shape[0] > 0:
        df = df.half()

    # 모델을 통해 텐서 예측을 수행한다.
    # 만약 Tensor의 크기가 0이면 y, y_val에 각각 0과 0.5를 대입한다.
    real_proba, fake_proba = (
        pred_vid(df, model)
        if df.shape[0] > 0
        else (torch.tensor(0).item(), torch.tensor(0.5).item())
    )
    # 1이면 fake, 0이면 real
    y = 1 if fake_proba >= real_proba else 0

    result = {
        "name": os.path.basename(file_path),
        "pred": y,
        "pred_proba": [real_proba, fake_proba]
    }

    return result


# Image prediction for a single image
def predict_image(
    file_path: str,
    model,
    fp16_mode: bool = False,
):

    # Load one RGB image
    try:
        im = Image.open(file_path).convert('RGB')
        arr = np.asarray(im)
    except Exception as e:
        print(f"Failed to open image {file_path}: {e}")
        arr = None

    # Detect face(s) and build a (N, 3, 224, 224) tensor
    if arr is not None:
        face, count = face_rec([arr])
        # save_detected_faces_png_named(
        #     face, count, file_path, "face_detect_images")
        df = preprocess_frame(
            face) if count > 0 else torch.empty((0, 3, 224, 224))
    else:
        df = torch.empty((0, 3, 224, 224))

    if fp16_mode and df.shape[0] > 0:
        df = df.half()

    # If no faces are detected, follow the same default as videos
    real_proba, fake_proba = (
        pred_vid(df, model)
        if df.shape[0] > 0
        else (torch.tensor(0).item(), torch.tensor(0.5).item())
    )
    y = 1 if fake_proba >= real_proba else 0
    
    result = {
        "name": os.path.basename(file_path),
        "pred": y,
        "pred_proba": [real_proba, fake_proba]
    }
    return result


def predict_path(
    file_path: str,
    model,
    fp16_mode: bool = False,
    num_frames: int = 15,
):
    if is_video(file_path):
        return predict_video(file_path, model, fp16_mode=fp16_mode, num_frames=num_frames)
    else:
        return predict_image(file_path, model, fp16_mode=fp16_mode)


def gen_parser():
    parser = argparse.ArgumentParser("GenConViT prediction")
    parser.add_argument("--p", type=str, help="video or image path")
    parser.add_argument(
        "--f", type=int, help="number of frames to process for prediction"
    )
    parser.add_argument(
        "--s", help="model size type: tiny, large.",
    )

    parser.add_argument("--fp16", type=str, help="half precision support")

    args = parser.parse_args()
    path = args.p
    num_frames = args.f if args.f else 15
    fp16 = True if args.fp16 else False

    net = 'genconvit'
    ed_weight = 'genconvit_ed_inference'
    vae_weight = 'genconvit_vae_inference'

    if args.s:
        if args.s in ['tiny', 'large']:
            config["model"]["backbone"] = f"convnext_{args.s}"
            config["model"]["embedder"] = f"swin_{args.s}_patch4_window7_224"
            config["model"]["type"] = args.s

    return path, num_frames, net, fp16, ed_weight, vae_weight


def main():
    # 제공된 하이퍼 파라미터 정의
    path, num_frames, net, fp16, ed_weight, vae_weight = gen_parser()

    # 모델을 한 번만 로드
    model = load_genconvit(config, net, ed_weight, vae_weight, fp16)
    print(f"Load {net} network" if net is not None else "Load ed and vae network")

    # 입력 경로에서 이미지/비디오 수집
    root_dir = path or "data"  # default path
    all_files = get_files_paths(
        root_dir, exs=['png', 'jpg', 'jpeg', 'mp4']
    )

    results = []
    print("Start Evaluating...")
    for file_path in tqdm(all_files, total=len(all_files)):
        result = predict_path(
            file_path=file_path,
            model=model,
            fp16_mode=fp16,
            num_frames=num_frames,
        )
        is_fake = result['pred'] == 1
        # print(
        #     f"File {result['name']}: {'FAKE' if is_fake else 'REAL'} for {result['pred_proba'][1] if is_fake else result['pred_proba'][0]}"
        # )
        results.append(result)

    # 결과 저장
    file_path = os.path.join("submission.csv")
    save_results_to_csv(results, file_path, for_submit=True)


if __name__ == "__main__":
    main()
