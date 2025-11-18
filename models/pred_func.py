import os
import numpy as np
import cv2
import torch
import dlib
from pathlib import Path
import face_recognition
from torchvision import transforms
from data_loader import normalize_data
from .config import load_config
from .genconvit import GenConViT
from decord import VideoReader, cpu
import glob
from PIL import Image
from typing import List

device = "cuda" if torch.cuda.is_available() else "cpu"


def save_detected_faces_png_named(
    faces: np.ndarray,
    count: int,
    src_filename: str,
    output_dir: str,
    start_index: int = 0,
) -> List[str]:
    """
    faces: (N, 224, 224, 3) RGB uint8 (face_rec 반환값)
    count: face_rec가 반환한 실제 얼굴 수
    src_filename: 원본 파일명(경로 포함 가능). 확장자를 제거한 뒤 이름을 만듦
    output_dir: 저장 폴더
    start_index: 동영상 프레임 번호 시작값(기본 0)

    규칙:
      - 이미지(단일 프레임/count==1): {filename}_detection.png
      - 동영상(복수 프레임/count>1): {filename}_detection_{i}.png
    반환: 저장된 파일 경로 리스트
    """
    os.makedirs(output_dir, exist_ok=True)
    if count <= 0 or faces.size == 0:
        return []

    # 확장자 제거한 기본 파일명
    base = Path(src_filename).name
    base_no_ext = base.rsplit(".", 1)[0] if "." in base else base

    saved_paths: List[str] = []
    limit = min(count, faces.shape[0])

    def _save_one(img_arr: np.ndarray, out_name: str) -> str:
        if img_arr.dtype != np.uint8:
            img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_arr, mode="RGB")
        out_path = os.path.join(output_dir, out_name)
        img.save(out_path, format="PNG", optimize=True)
        return out_path

    if limit == 1:
        out_name = f"{base_no_ext}_detection.png"
        saved_paths.append(_save_one(faces[0], out_name))
    else:
        for i in range(limit):
            out_name = f"{base_no_ext}_detection_{start_index + i}.png"
            saved_paths.append(_save_one(faces[i], out_name))

    return saved_paths


def empty_video_batch():
    # (0, 3, 224, 224) empty tensor on the current device, matching preprocess_frame's dtype
    return torch.empty((0, 3, 224, 224), device=device, dtype=torch.float32)


def load_genconvit(config, net, ed_weight, vae_weight, fp16):
    model = GenConViT(
        config,
        ed=ed_weight,
        vae=vae_weight,
        net=net,
        fp16=fp16
    )

    model.to(device)
    model.eval()
    if fp16:
        model.half()

    return model


def face_rec(frames, p=None, klass=None):
    temp_face = np.zeros((len(frames), 224, 224, 3), dtype=np.uint8)
    count = 0
    # mod = "cnn" if dlib.DLIB_USE_CUDA else "hog"
    mod = "hog"  # dlib 버전이 틀려서 cuda 호환 X. 일단 이렇게 처리.

    for _, frame in enumerate(frames):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        face_locations = face_recognition.face_locations(
            frame, number_of_times_to_upsample=0, model=mod
        )

        for face_location in face_locations:
            top, right, bottom, left = face_location
            # Expand the box, then clip to valid image bounds
            pad = 24
            h, w = frame.shape[:2]
            top = max(0, top - pad)
            left = max(0, left - pad)
            bottom = min(h, bottom + pad)
            right = min(w, right + pad)

            # Skip invalid/empty regions after clipping
            if bottom <= top or right <= left:
                continue

            # Crop safely
            face_image = frame[top:bottom, left:right]
            if face_image.size == 0:
                continue

            # Resize and convert to RGB
            face_image = cv2.resize(
                face_image, (224, 224), interpolation=cv2.INTER_AREA)
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

            if count < len(frames):
                temp_face[count] = face_image
                count += 1
            else:
                break

    return ([], 0) if count == 0 else (temp_face[:count], count)


def preprocess_frame(frame):
    df_tensor = torch.tensor(frame, device=device).float()
    df_tensor = df_tensor.permute((0, 3, 1, 2))

    # Lookup the transform once instead of inside the loop for each frame
    vid_norm = normalize_data()["vid"]
    for i in range(len(df_tensor)):
        df_tensor[i] = vid_norm(df_tensor[i] / 255.0)

    return df_tensor


def pred_vid(df, model):
    with torch.no_grad():
        return max_prediction_value(torch.sigmoid(model(df).squeeze()))


def max_prediction_value(y_pred):  # y_pred: Tensor(frame_length, 2)
    """
    모델은 0: fake, 1:real로 예측한다. 하지만 이는 관행(+경진대회)에 부적합함으로 0: real, 1:fake로 처리한다.
    이에 따라 본 함수는 (real일 확률, fake일 확률)로 2개의 값을 가지는 tuple을 반환한다.
    각 값은 0과 1 사이의 값을 가지며 각 상태에 대한 확률을 정의한다.
    """
    # 0과 1에 대하여 각 프레임별 평균값을 구한다.
    # 만약 frame_length=1일 경우 squeeze를 통해 2차원이 아닌 1차원이 될 수 있음으로 복원한다.
    if y_pred.dim() == 1:
        y_pred = y_pred.unsqueeze(0)
    mean_val = torch.mean(y_pred, dim=0)

    return (mean_val[1].item(), mean_val[0].item())


def real_or_fake(prediction):
    return {0: "REAL", 1: "FAKE"}[prediction ^ 1]


def extract_frames(video_file, num_frames=15, consecutive=True):
    vr = VideoReader(video_file, ctx=cpu(0))
    total_frames = len(vr)

    if num_frames == -1:
        # if -1, get all frames
        indices = np.arange(total_frames).astype(int)
    else:
        if consecutive:
            # Extract consecutive frames from the middle of the video
            # This preserves temporal consistency for better fake detection
            start_idx = max(0, (total_frames - num_frames) // 2)
            end_idx = min(total_frames, start_idx + num_frames)
            indices = np.arange(start_idx, end_idx, dtype=int)
        else:
            # Original: evenly spaced frames across the video
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    return vr.get_batch(indices).asnumpy()  # seek frames with step_size


def df_face_from_folder(vid, num_frames):
    img_list = glob.glob(vid + "/*")
    img = []
    for f in img_list:
        try:
            im = Image.open(f).convert('RGB')
            img.append(np.asarray(im))
        except:
            pass

    face, count = face_rec(img[:num_frames])
    return preprocess_frame(face) if count > 0 else empty_video_batch()


def df_face(file_path, num_frames, consecutive=True):
    img = extract_frames(file_path, num_frames, consecutive=consecutive)
    face, count = face_rec(img)
    # save_detected_faces_png_named(
    #     face, count, file_path, "face_detect_images")
    return preprocess_frame(face) if count > 0 else empty_video_batch()


def is_video(vid):
    return os.path.isfile(vid) and vid.endswith(
        tuple([".avi", ".mp4", ".mpg", ".mpeg", ".mov"])
    )


def is_video_folder(vid_folder):
    img_list = glob.glob(vid_folder + "/*")
    return len(img_list) >= 1 and img_list[0].endswith(tuple(["png", "jpeg", "jpg"]))


def set_result():
    return {
        "video": {
            "name": [],
            "pred": [],
            "klass": [],
            "pred_label": [],
            "correct_label": [],
        }
    }


def store_result(
    result, filename, y, y_val, klass, correct_label=None, compression=None
):
    result["video"]["name"].append(filename)
    result["video"]["pred"].append(y_val)
    result["video"]["klass"].append(klass.lower())
    result["video"]["pred_label"].append(real_or_fake(y))

    if correct_label is not None:
        result["video"]["correct_label"].append(correct_label)

    if compression is not None:
        result["video"]["compression"].append(compression)

    return result
