import os
import argparse
import torch
from models.pred_func import load_genconvit, face_rec, preprocess_frame, is_video, extract_frames
from models.config import load_config
from typing import List, Dict, Union
import csv
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
import cv2
import multiprocessing

config = load_config()


# =====================================================================
# CPU WORKER FUNCTION FOR PARALLEL PROCESSING
# =====================================================================

def process_single_file_cpu(args):
    """CPU 작업: 프레임 추출 및 얼굴 검출"""
    file_path, num_frames = args
    filename = os.path.basename(file_path)
    
    try:
        if is_video(file_path):
            # Extract frames
            raw_frames = extract_frames(file_path, num_frames, consecutive=False)
            
            # Detect faces
            face_crops, count = face_rec(raw_frames)
            
            if count > 0:
                # Calculate Laplacian variance for each frame
                laplacian_vars = []
                for face in face_crops[:count]:
                    if len(face.shape) == 3:
                        gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
                    else:
                        gray = face
                    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                    laplacian_vars.append(laplacian.var())
                
                return (filename, face_crops[:count], raw_frames[:count], laplacian_vars, True, None)
            else:
                return (filename, None, None, [], True, None)
        else:
            # Load image
            im = Image.open(file_path).convert('RGB')
            arr = np.asarray(im)
            
            # Detect face
            face, count = face_rec([arr])
            
            if count > 0:
                face_crop = face[0]
                if len(face_crop.shape) == 3:
                    gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
                else:
                    gray = face_crop
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                lap_var = laplacian.var()
                
                return (filename, face[:count], [arr], [lap_var], False, None)
            else:
                return (filename, None, None, [], False, None)
    
    except Exception as e:
        return (filename, None, None, [], None, str(e))


# =====================================================================
# TEMPORAL ANALYSIS
# =====================================================================

def calculate_temporal_variance(frames_np):
    """프레임 간 optical flow의 분산 계산"""
    if len(frames_np) < 2:
        return 0.0
    
    target_size = (224, 224)
    normalized_frames = []
    
    for frame in frames_np:
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame
        
        if gray.shape[:2] != target_size:
            gray = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
        
        normalized_frames.append(gray)
    
    prev_gray = None
    magnitudes = []
    
    for gray in normalized_frames:
        if prev_gray is not None:
            try:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None, 
                    pyr_scale=0.5, levels=3, winsize=15, 
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                )
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                magnitudes.append(np.mean(mag))
            except cv2.error:
                continue
        
        prev_gray = gray
    
    return np.var(magnitudes) if magnitudes else 0.0


def calculate_detail_stability(laplacian_vars):
    """프레임 간 디테일(선명도)의 일관성 측정"""
    if len(laplacian_vars) < 2:
        return 0.0
    return np.std(laplacian_vars)


def classify_video_type(temporal_var, detail_stability, avg_sharpness):
    """비디오 타입 분류"""
    TEMPORAL_HIGH = 0.03
    TEMPORAL_LOW = 0.01
    DETAIL_HIGH = 50.0
    DETAIL_LOW = 30.0
    SHARP_THRESHOLD = 150.0
    
    if avg_sharpness > SHARP_THRESHOLD and temporal_var > TEMPORAL_HIGH:
        return True, 0.85, "VEO3-type"
    
    if avg_sharpness < SHARP_THRESHOLD and temporal_var < TEMPORAL_LOW and detail_stability > DETAIL_HIGH:
        return True, 0.80, "SORA2-type"
    
    if detail_stability > DETAIL_HIGH and temporal_var > TEMPORAL_HIGH:
        return True, 0.75, "Hybrid-type"
    
    if detail_stability < DETAIL_LOW and TEMPORAL_LOW < temporal_var < TEMPORAL_HIGH:
        return False, 0.90, "Real-type"
    
    return None, 0.5, "Uncertain-type"


# =====================================================================
# ARTIFACT DETECTION
# =====================================================================

def check_frequency_artifacts(image):
    """주파수 도메인에서 deepfake artifacts 검사"""
    if isinstance(image, Image.Image):
        image = np.array(image.convert('L'))
    
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_shift)
    
    h, w = magnitude_spectrum.shape
    center_h, center_w = h // 2, w // 2
    
    high_freq_region = magnitude_spectrum.copy()
    mask_radius = min(h, w) // 4
    y, x = np.ogrid[:h, :w]
    mask = (x - center_w)**2 + (y - center_h)**2 <= mask_radius**2
    high_freq_region[mask] = 0
    
    total_energy = np.sum(magnitude_spectrum)
    high_freq_energy = np.sum(high_freq_region)
    high_freq_ratio = high_freq_energy / (total_energy + 1e-10)
    
    return high_freq_ratio


def check_face_boundary_artifacts(image):
    """얼굴 경계선 주변의 artifacts 검사"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    h, w = image.shape[:2]
    border_width = int(min(h, w) * 0.1)
    
    top_border = image[:border_width, :]
    bottom_border = image[-border_width:, :]
    left_border = image[:, :border_width]
    right_border = image[:, -border_width:]
    
    border_regions = [top_border, bottom_border, left_border, right_border]
    border_stds = [np.std(region) for region in border_regions]
    avg_border_std = np.mean(border_stds)
    
    center_region = image[border_width:-border_width, border_width:-border_width]
    center_std = np.std(center_region)
    
    boundary_anomaly = abs(avg_border_std - center_std) / (center_std + 1e-10)
    return boundary_anomaly


def check_color_consistency(image):
    """색상 일관성 검사"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    r_mean = np.mean(image[:,:,0])
    g_mean = np.mean(image[:,:,1])
    b_mean = np.mean(image[:,:,2])
    channel_imbalance = np.std([r_mean, g_mean, b_mean]) / (np.mean([r_mean, g_mean, b_mean]) + 1e-10)
    
    return channel_imbalance


def perform_secondary_checks(image, laplacian_var):
    """선명한 이미지에 대한 추가 검증"""
    freq_artifact = check_frequency_artifacts(image)
    boundary_artifact = check_face_boundary_artifacts(image)
    color_inconsistency = check_color_consistency(image)
    
    freq_score = 0.0
    if freq_artifact < 0.01 or freq_artifact > 0.12:
        freq_score = min(abs(freq_artifact - 0.05) / 0.05, 1.0)
    
    boundary_score = min(boundary_artifact / 1.0, 1.0)
    color_score = min(color_inconsistency / 0.5, 1.0)
    
    weights = {'freq': 0.5, 'boundary': 0.2, 'color': 0.3}
    
    suspicion_score = (
        weights['freq'] * freq_score +
        weights['boundary'] * boundary_score +
        weights['color'] * color_score
    )
    
    return suspicion_score


# =====================================================================
# CSV SAVE FUNCTIONS
# =====================================================================

def save_results_to_csv(results: List[Dict], filepath: str, for_submit=False) -> None:
    """최종 제출용 CSV 저장"""
    sorted_results = sorted(results, key=lambda x: x['name'])
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if for_submit:
            w.writerow(("filename", "label"))
            for r in sorted_results:
                w.writerow((r['name'], r['pred']))
        else:
            w.writerow(("name", "pred", "pred_proba"))
            for r in sorted_results:
                w.writerow((r["name"], r["pred"], r["pred_proba"]))


def save_frame_probabilities_to_csv(results: List[Dict], filepath: str) -> None:
    """프레임별 확률 CSV 저장"""
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "frame_idx", "real_proba", "fake_proba", "prediction"])
        for r in results:
            w.writerow([r['filename'], r['frame_idx'], r['real_proba'], r['fake_proba'], r['pred']])


def get_files_paths(main_path: str, exs: List[str]) -> List[str]:
    """파일 경로 수집"""
    exs = {("." + ext.lstrip(".")).lower() for ext in exs}
    results = []

    for root, _, files in os.walk(main_path):
        for fname in files:
            abs_path = os.path.abspath(os.path.join(root, fname))
            _, ext = os.path.splitext(fname)
            if ext.lower() in exs:
                results.append(abs_path)

    return results


# =====================================================================
# PREDICTION FUNCTIONS WITH POSTPROCESSING
# =====================================================================

def predict_video(
    file_path: str,
    model,
    fp16_mode: bool = False,
    num_frames: int = 15,
):
    """비디오 예측 with 강력한 postprocessing"""
    filename = os.path.basename(file_path)
    
    # Extract frames
    raw_frames = extract_frames(file_path, num_frames, consecutive=False)
    
    # Detect faces
    face_crops, count = face_rec(raw_frames)
    
    frame_results = []
    
    if count > 0:
        # Preprocess
        df = preprocess_frame(face_crops[:count])
        
        if fp16_mode and df.shape[0] > 0:
            df = df.half()
        
        if df.shape[0] > 0:
            with torch.no_grad():
                outputs = model(df)
                probs = torch.softmax(outputs, dim=1)
                
                real_probas = probs[:, 0].cpu().numpy()
                fake_probas = probs[:, 1].cpu().numpy()
                
                # Store per-frame results
                for frame_idx, (real_p, fake_p) in enumerate(zip(real_probas, fake_probas)):
                    pred = 1 if fake_p >= real_p else 0
                    frame_results.append({
                        'filename': filename,
                        'frame_idx': frame_idx,
                        'real_proba': float(real_p),
                        'fake_proba': float(fake_p),
                        'pred': pred
                    })
            
            # Calculate average
            avg_real = np.mean(real_probas)
            avg_fake = np.mean(fake_probas)
            predicted_class = 1 if avg_fake >= avg_real else 0
            
            # ===== POSTPROCESSING: 더 공격적으로 FAKE 탐지 =====
            
            # 1. Sequence analysis - 임계값 낮춤 (더 민감하게)
            if df.shape[0] > 5:
                window_size = 5
                moving_avg = np.convolve(fake_probas, np.ones(window_size)/window_size, mode="valid")
                
                high_fake_ratio = np.mean(moving_avg > 0.45)  # 0.55 -> 0.45
                mid_fake_ratio = np.mean(moving_avg > 0.35)   # 0.4 -> 0.35
                max_moving_avg = np.max(moving_avg)
                avg_fake_prob = np.mean(fake_probas)
                
                # 더 낮은 임계값으로 FAKE 판정
                if (high_fake_ratio > 0.12 or mid_fake_ratio > 0.30 or 
                    max_moving_avg > 0.55 or avg_fake_prob > 0.30):
                    predicted_class = 1
                    print(f"[SEQ-OVERRIDE] {filename}: high={high_fake_ratio:.2f}, mid={mid_fake_ratio:.2f}, max={max_moving_avg:.2f}, avg={avg_fake_prob:.2f}")
            
            # 2. 프레임 간 변동성 체크 - FAKE 프레임이 하나라도 많으면 의심
            fake_frame_count = np.sum(fake_probas > 0.5)
            fake_frame_ratio = fake_frame_count / len(fake_probas)
            
            if fake_frame_ratio > 0.25:  # 25% 이상이 fake면 전체를 fake로
                predicted_class = 1
                print(f"[FRAME-RATIO-OVERRIDE] {filename}: {fake_frame_ratio:.2%} frames are fake")
            
            # 3. Temporal analysis
            if len(raw_frames) >= 5:
                # Calculate Laplacian variance
                laplacian_vars = []
                for face in face_crops[:count]:
                    if len(face.shape) == 3:
                        gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
                    else:
                        gray = face
                    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                    laplacian_vars.append(laplacian.var())
                
                temporal_var = calculate_temporal_variance(raw_frames)
                detail_stability = calculate_detail_stability(laplacian_vars)
                avg_sharpness = np.mean(laplacian_vars)
                
                is_fake_temporal, temp_confidence, video_type = classify_video_type(
                    temporal_var, detail_stability, avg_sharpness
                )
                
                if is_fake_temporal is not None and is_fake_temporal:
                    predicted_class = 1
                    print(f"[TEMPORAL-OVERRIDE] {filename}: {video_type}")
            
            result = {
                "name": filename,
                "pred": predicted_class,
                "pred_proba": [float(avg_real), float(avg_fake)],
                "frame_results": frame_results
            }
        else:
            result = {
                "name": filename,
                "pred": 0,
                "pred_proba": [0.5, 0.5],
                "frame_results": []
            }
    else:
        result = {
            "name": filename,
            "pred": 0,
            "pred_proba": [0.5, 0.5],
            "frame_results": []
        }
    
    return result


def predict_image(
    file_path: str,
    model,
    fp16_mode: bool = False,
):
    """이미지 예측 with postprocessing"""
    filename = os.path.basename(file_path)
    
    try:
        im = Image.open(file_path).convert('RGB')
        arr = np.asarray(im)
    except Exception as e:
        print(f"Failed to open image {file_path}: {e}")
        return {
            "name": filename,
            "pred": 0,
            "pred_proba": [0.5, 0.5],
            "frame_results": []
        }
    
    # Detect face
    face, count = face_rec([arr])
    
    if count > 0:
        df = preprocess_frame(face[:count])
        
        if fp16_mode and df.shape[0] > 0:
            df = df.half()
        
        if df.shape[0] > 0:
            with torch.no_grad():
                outputs = model(df)
                probs = torch.softmax(outputs, dim=1)
                
                real_proba = probs[0, 0].cpu().item()
                fake_proba = probs[0, 1].cpu().item()
                predicted_class = 1 if fake_proba >= real_proba else 0
                
                # ===== POSTPROCESSING: Secondary checks for high sharpness =====
                face_crop = face[0]
                if len(face_crop.shape) == 3:
                    gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
                else:
                    gray = face_crop
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                lap_var = laplacian.var()
                
                SHARPNESS_THRESHOLD = 180.0
                
                if lap_var > SHARPNESS_THRESHOLD and predicted_class == 0:
                    suspicion_score = perform_secondary_checks(face_crop, lap_var)
                    
                    SUSPICION_THRESHOLD = 0.60
                    
                    if suspicion_score > SUSPICION_THRESHOLD:
                        predicted_class = 1
                        print(f"[OVERRIDE] {filename}: Sharp but suspicious (susp={suspicion_score:.3f}, lap={lap_var:.1f})")
                
                frame_results = [{
                    'filename': filename,
                    'frame_idx': 0,
                    'real_proba': float(real_proba),
                    'fake_proba': float(fake_proba),
                    'pred': predicted_class
                }]
                
                result = {
                    "name": filename,
                    "pred": predicted_class,
                    "pred_proba": [float(real_proba), float(fake_proba)],
                    "frame_results": frame_results
                }
        else:
            result = {
                "name": filename,
                "pred": 0,
                "pred_proba": [0.5, 0.5],
                "frame_results": []
            }
    else:
        result = {
            "name": filename,
            "pred": 0,
            "pred_proba": [0.5, 0.5],
            "frame_results": []
        }
    
    return result


def predict_path(
    file_path: str,
    model,
    fp16_mode: bool = False,
    num_frames: int = 15,
):
    """파일 타입에 따라 예측"""
    if is_video(file_path):
        return predict_video(file_path, model, fp16_mode=fp16_mode, num_frames=num_frames)
    else:
        return predict_image(file_path, model, fp16_mode=fp16_mode)


def gen_parser():
    parser = argparse.ArgumentParser("GenConViT prediction with postprocessing")
    parser.add_argument("--p", type=str, help="video or image path", default="data")
    parser.add_argument("--f", type=int, help="number of frames to process", default=30)
    parser.add_argument("--s", help="model size type: tiny, large.")
    parser.add_argument("--fp16", type=str, help="half precision support")

    args = parser.parse_args()
    path = args.p
    num_frames = args.f
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
    # 파라미터 로드
    path, num_frames, net, fp16, ed_weight, vae_weight = gen_parser()

    # 모델 로드
    print("Loading model...")
    model = load_genconvit(config, net, ed_weight, vae_weight, fp16)
    print(f"✓ Loaded {net} network\n")

    # 파일 수집
    root_dir = path
    all_files = get_files_paths(root_dir, exs=['png', 'jpg', 'jpeg', 'mp4'])
    print(f"✓ Found {len(all_files)} files")

    # Setup multiprocessing
    num_workers = min(max(1, multiprocessing.cpu_count() - 1), 8)
    print(f"✓ Using {num_workers} worker processes for preprocessing\n")
    
    # Prepare worker arguments
    worker_args = [(file_path, num_frames) for file_path in all_files]

    results = []
    all_frame_results = []
    
    # Statistics
    sequence_override_count = 0
    frame_ratio_override_count = 0
    temporal_override_count = 0
    secondary_check_count = 0
    
    print("Start Evaluating with Postprocessing...\n")
    
    # Parallel preprocessing (CPU)
    with multiprocessing.Pool(processes=num_workers) as pool:
        with tqdm(total=len(all_files), desc="Processing files") as pbar:
            for result in pool.imap_unordered(process_single_file_cpu, worker_args):
                filename, face_crops, raw_frames, laplacian_vars, is_video_file, error = result
                
                # Check for errors
                if error:
                    print(f"Error processing {filename}: {error}")
                    results.append({
                        "name": filename,
                        "pred": 0,
                        "pred_proba": [0.5, 0.5]
                    })
                    pbar.update(1)
                    continue
                
                # No face detected
                if face_crops is None:
                    results.append({
                        "name": filename,
                        "pred": 0,
                        "pred_proba": [0.5, 0.5]
                    })
                    pbar.update(1)
                    continue
                
                # ===== GPU INFERENCE =====
                df = preprocess_frame(face_crops)
                if fp16 and df.shape[0] > 0:
                    df = df.half()
                
                if df.shape[0] > 0:
                    with torch.no_grad():
                        outputs = model(df)
                        probs = torch.softmax(outputs, dim=1)
                        
                        real_probas = probs[:, 0].cpu().numpy()
                        fake_probas = probs[:, 1].cpu().numpy()
                        
                        # Store per-frame results
                        for frame_idx, (real_p, fake_p) in enumerate(zip(real_probas, fake_probas)):
                            pred = 1 if fake_p >= real_p else 0
                            all_frame_results.append({
                                'filename': filename,
                                'frame_idx': frame_idx,
                                'real_proba': float(real_p),
                                'fake_proba': float(fake_p),
                                'pred': pred
                            })
                    
                    # Calculate average
                    avg_real = np.mean(real_probas)
                    avg_fake = np.mean(fake_probas)
                    predicted_class = 1 if avg_fake >= avg_real else 0
                    
                    # ===== POSTPROCESSING =====
                    
                    if is_video_file:
                        # 1. Sequence analysis
                        if df.shape[0] > 5:
                            window_size = 5
                            moving_avg = np.convolve(fake_probas, np.ones(window_size)/window_size, mode="valid")
                            
                            high_fake_ratio = np.mean(moving_avg > 0.45)
                            mid_fake_ratio = np.mean(moving_avg > 0.35)
                            max_moving_avg = np.max(moving_avg)
                            avg_fake_prob = np.mean(fake_probas)
                            
                            if (high_fake_ratio > 0.12 or mid_fake_ratio > 0.30 or 
                                max_moving_avg > 0.55 or avg_fake_prob > 0.30):
                                if predicted_class == 0:
                                    sequence_override_count += 1
                                predicted_class = 1
                                # print(f"[SEQ-OVERRIDE] {filename}")
                        
                        # 2. Frame ratio check
                        fake_frame_count = np.sum(fake_probas > 0.5)
                        fake_frame_ratio = fake_frame_count / len(fake_probas)
                        
                        if fake_frame_ratio > 0.25:
                            if predicted_class == 0:
                                frame_ratio_override_count += 1
                            predicted_class = 1
                            # print(f"[FRAME-RATIO-OVERRIDE] {filename}: {fake_frame_ratio:.2%}")
                        
                        # 3. Temporal analysis
                        if len(raw_frames) >= 5 and laplacian_vars:
                            temporal_var = calculate_temporal_variance(raw_frames)
                            detail_stability = calculate_detail_stability(laplacian_vars)
                            avg_sharpness = np.mean(laplacian_vars)
                            
                            is_fake_temporal, temp_confidence, video_type = classify_video_type(
                                temporal_var, detail_stability, avg_sharpness
                            )
                            
                            if is_fake_temporal is not None and is_fake_temporal:
                                if predicted_class == 0:
                                    temporal_override_count += 1
                                predicted_class = 1
                                # print(f"[TEMPORAL-OVERRIDE] {filename}: {video_type}")
                    
                    else:
                        # Image postprocessing
                        if laplacian_vars:
                            lap_var = laplacian_vars[0]
                            SHARPNESS_THRESHOLD = 180.0
                            
                            if lap_var > SHARPNESS_THRESHOLD and predicted_class == 0:
                                suspicion_score = perform_secondary_checks(face_crops[0], lap_var)
                                SUSPICION_THRESHOLD = 0.60
                                
                                if suspicion_score > SUSPICION_THRESHOLD:
                                    predicted_class = 1
                                    secondary_check_count += 1
                                    # print(f"[OVERRIDE] {filename}: Sharp but suspicious")
                    
                    results.append({
                        "name": filename,
                        "pred": predicted_class,
                        "pred_proba": [float(avg_real), float(avg_fake)]
                    })
                else:
                    results.append({
                        "name": filename,
                        "pred": 0,
                        "pred_proba": [0.5, 0.5]
                    })
                
                pbar.update(1)

    # 결과 저장
    print("\n✓ Processing complete!\n")
    
    # 1. 최종 제출용 CSV
    save_results_to_csv(results, "submission.csv", for_submit=True)
    print(f"✓ Saved submission to submission.csv")
    
    # # 2. 프레임별 확률 CSV
    # if all_frame_results:
    #     save_frame_probabilities_to_csv(all_frame_results, "frame_probabilities.csv")
    #     print(f"✓ Saved frame probabilities to frame_probabilities.csv")
    
    # 통계 출력
    print(f"\n=== Detection Statistics ===")
    print(f"Sequence analysis overrides: {sequence_override_count}")
    print(f"Frame ratio overrides: {frame_ratio_override_count}")
    print(f"Temporal analysis overrides: {temporal_override_count}")
    print(f"Secondary check overrides: {secondary_check_count}")
    
    num_fake = sum(1 for r in results if r['pred'] == 1)
    num_real = len(results) - num_fake
    
    print(f"\n=== Prediction Summary ===")
    print(f"Total files: {len(results)}")
    print(f"  Real: {num_real} ({num_real/len(results)*100:.1f}%)")
    print(f"  Fake: {num_fake} ({num_fake/len(results)*100:.1f}%)")
    
    if all_frame_results:
        num_fake_frames = sum(1 for r in all_frame_results if r['pred'] == 1)
        print(f"\nTotal frames: {len(all_frame_results)}")
        print(f"  Fake frames: {num_fake_frames} ({num_fake_frames/len(all_frame_results)*100:.1f}%)")


if __name__ == "__main__":
    main()