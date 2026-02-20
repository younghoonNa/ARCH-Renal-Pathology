    # -*- coding: utf-8 -*-
    """    ===========================================================================
               PATHOLOGY WSI ANALYSIS PIPELINE (End-to-End)
===========================================================================

[개요]
이 파이프라인은 WSI(Whole Slide Image) 파일을 입력받아 Segmentation, 특징 추출,
사구체 분류(M0/M1)를 거쳐 최종 진단 결과를 예측하는 전체 공정입니다.

[경로 설정]
1. 작업 루트 (ROOT_DIR): /home/khdp-user/workspace/
2. 코드 위치 (Script DIR): /home/khdp-user/workspace/Final_submission/
   (* 모든 파이썬 스크립트 6개와 메인 주피터 노트북은 이 폴더에 위치해야 합니다)
3. 모델 위치 (Inference): /home/khdp-user/workspace/Final_submission/TransMIL_pt/
   (* 5개의 fold별 모델 파일이 이 폴더 안에 있어야 합니다)

[디렉토리 구조]
/home/khdp-user/workspace/
 |-- dataset/
 |    |-- Slide/                   <-- [입력] 새로운 SVS 파일을 여기에 넣으세요.
 |    |-- CSV/GT_label.csv         <-- 데이터셋 리스트
 |    `-- Models/                  <-- Segmentation 모델 (.pt)
 |-- Annotation_3_final/           <-- [중간산출] XML 좌표 (Glomerulus, IFTA)
 |-- Patch_3_final/                <-- [중간산출] 일반 패치 H5 (20x)
 |-- Patch_3_Glom_final/           <-- [중간산출] 사구체 크롭 H5
 |-- GigaPath/
 |    |-- ..._Normalized/          <-- [중간산출] 일반 특징 추출 결과
 |    |-- ..._M0M1C/               <-- [중간산출] 사구체 M0/M1 분류 결과
 |    `-- ..._Merged/              <-- [중간산출] 최종 병합된 특징 데이터
 `-- Final_submission/             <-- [실행 위치]
      |-- 01_segmentation.py
      |-- 02_glom_generation.py
      |-- 03_general_features.py
      |-- 04_glom_classification.py
      |-- 05_merge_h5.py
      |-- 06_inference.py
      |-- Main_Pipeline.ipynb      <-- [실행 파일] 주피터 노트북
      |-- infer_pred.csv           <-- [최종결과] 생성된 예측 파일
      `-- TransMIL_pt/             <-- [모델폴더] 5개의 모델 가중치 파일

===========================================================================
[실행 순서]
===========================================================================

Step 1. 전처리 및 영역 분할 (Segmentation & Patching)
   - 파일명: 01_segmentation.py
   - 기  능: SVS에서 조직을 찾고 UNet으로 Glomerulus/IFTA 분할 후 XML 및 기본 H5 저장
   - 입  력: dataset/Slide/*.svs
   - 출  력: Annotation_3_final/*.xml, Patch_3_final/*.h5

Step 2. 사구체 영역 정밀 크롭 (Glomerulus Cropping)
   - 파일명: 02_glom_generation.py
   - 기  능: 생성된 XML 좌표를 기반으로 사구체 이미지만 정밀하게 잘라냄
   - 입  력: Annotation_3_final/*.xml + dataset/Slide/*.svs
   - 출  력: Patch_3_Glom_final/*.h5

Step 3. 일반 조직 특징 추출 (General Feature Extraction)
   - 파일명: 03_general_features.py
   - 기  능: Step 1의 일반 패치들을 GigaPath 모델로 임베딩 (Macenko 정규화 포함)
   - 입  력: Patch_3_final/*.h5
   - 출  력: GigaPath/x20_..._Normalized/*.h5

Step 4. 사구체 분류 및 특징 추출 (Glomerulus Classification M0/M1)
   - 파일명: 04_glom_classification.py
   - 기  능: Step 2의 사구체 패치를 M0(정상)/M1(경화)로 분류하고 임베딩 저장
   - 입  력: Patch_3_Glom_final/*.h5
   - 출  력: GigaPath/x20_..._M0M1C/*.h5

Step 5. 특징 데이터 병합 (Merge Features)
   - 파일명: 05_merge_h5.py
   - 기  능: Step 3(일반 조직)와 Step 4(분류된 사구체)의 H5 파일을 하나로 합침
   - 입  력: Step 3 출력물 + Step 4 출력물
   - 출  력: GigaPath/x20_..._Merged/*.h5

Step 6. 최종 진단 예측 (Final Inference)
   - 파일명: 06_inference.py
   - 기  능: 병합된 특징 데이터를 TransMIL 모델에 넣어 최종 확률 예측
   - 입  력: GigaPath/x20_..._Merged/*.h5, TransMIL_pt/*.pt
   - 출  력: Final_submission/infer_pred.csv

===========================================================================
[주의 사항]
===========================================================================
1. 경로 확인: 위 ROOT_DIR 경로가 실제 서버 환경과 일치하는지 확인하십시오.
2. GPU 사용: 모든 스크립트는 CUDA:0 (0번 GPU) 사용을 기본으로 합니다.
3. 파일 이름: SVS 파일명에 공백이나 특수문자가 없는지 확인하십시오.
4. 모델 확인: Final_submission/TransMIL_pt/ 폴더 안에 5개의 .pt 파일이 있는지 확인하십시오.

    NOTE:
    - HuggingFace 토큰(HF_TOKEN)은 코드에 하드코딩하지 않았습니다.
      필요하다면 실행 전에 환경변수로 주입하세요:
        export HF_TOKEN=...
    """

import os
import sys
import time
import gc
import csv
import numpy as np
import pandas as pd
import cv2
import torch
import openslide
import segmentation_models_pytorch as smp
import albumentations as A
import xml.etree.ElementTree as ET
from xml.dom import minidom
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import h5py

# 이미지 픽셀 제한 해제
Image.MAX_IMAGE_PIXELS = None 

# ============================================================
# [설정 및 경로]
# ============================================================
BASE_DIR = Path("/home/khdp-user/workspace")
DATASET_CSV = BASE_DIR / "dataset/CSV/GT_label.csv"
SVS_ROOT = BASE_DIR / "dataset/Slide"

# 저장 경로 (_final 추가)
ANNOTATION_GLOM_ROOT = BASE_DIR / "Annotation_3_final/Glomerulus"
ANNOTATION_IFTA_ROOT = BASE_DIR / "Annotation_3_final/IFTA"
SAVE_H5_ROOT = BASE_DIR / "Patch_3_final"
LOG_CSV_PATH = BASE_DIR / "processing_log_final.csv"

# 디렉토리 생성
for path in [ANNOTATION_GLOM_ROOT, ANNOTATION_IFTA_ROOT, SAVE_H5_ROOT]:
    path.mkdir(parents=True, exist_ok=True)

# 모델 경로
MODEL_PATH_GLOM = BASE_DIR / "dataset/Models/Glom_run_seg/best_model.pt"
MODEL_PATH_IFTA = BASE_DIR / "dataset/Models/IFTA_run_seg/best_model.pt"

# 하이퍼파라미터
GPU_ID = 0
DEVICE = f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu"

# 1. Segmentation 설정 (모델 추론용 - 10배율)
SEG_PATCH_SIZE = 512        
SEG_OVERLAP_RATIO = 0.2     
SEG_TARGET_MAG = 10.0       
SEG_THRESH_GLOM = 0.5
SEG_THRESH_IFTA = 0.5
GLOM_AREA_RATIO_THRESH = 0.04

# 2. H5 Extraction 설정 (최종 저장용 - 20배율)
H5_PATCH_SIZE = 224         
H5_TARGET_MAG = 20.0        
H5_TH_TISSUE = 0.25
H5_TH_GLOM = 0.1
H5_TH_IFTA = 0.5
H5_TH_EXTRA = 0.5           # [중요] Extrarenal Discard Threshold

# 3. ROI Search 설정
SEARCH_MAG = 1.25           
TISSUE_RATIO_THRESH = 0.25 

# 배치 사이즈
BATCH_SIZE = 256

# ============================================================
# [유틸리티 함수]
# ============================================================
def load_models():
    print(f"[-] 모델 로딩 중... (Device: {DEVICE})")
    models = {}
    
    # Glomerulus
    if os.path.exists(MODEL_PATH_GLOM):
        m_glom = smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=1, activation=None)
        m_glom.load_state_dict(torch.load(MODEL_PATH_GLOM, map_location=DEVICE))
        m_glom.to(DEVICE).eval()
        models['glom'] = m_glom
    else:
        print(f"[Error] Glom Model Not Found: {MODEL_PATH_GLOM}")
        sys.exit()

    # IFTA (Classes: 0=Bg, 1=IFTA, 2=Medulla, 3=Extrarenal)
    if os.path.exists(MODEL_PATH_IFTA):
        m_ifta = smp.Unet(encoder_name="resnet50", encoder_weights=None,  in_channels=3, classes=4, activation=None)
        m_ifta.load_state_dict(torch.load(MODEL_PATH_IFTA, map_location=DEVICE))
        m_ifta.to(DEVICE).eval()
        models['ifta'] = m_ifta
    else:
        print(f"[Error] IFTA Model Not Found: {MODEL_PATH_IFTA}")
        sys.exit()
        
    return models

def get_tissue_mask_robust(img_rgb):
    img_np = np.array(img_rgb)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    val, tissue_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel) 
    return tissue_mask

def save_xml(slide_name, contours_dict, save_dir, mpp):
    root = ET.Element("Annotations", MicronsPerPixel=f"{mpp:.6f}")
    
    for layer_name, data in contours_dict.items():
        layer_id = data['id']
        color = data['color']
        contours = data['contours']
        
        anno = ET.SubElement(root, "Annotation", Id=layer_id, Name=layer_name, ReadOnly="0", Type="4", Visible="1", LineColor=color)
        regions = ET.SubElement(anno, "Regions")
        
        for idx, contour in enumerate(contours):
            epsilon = 0.001 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) < 3: continue
            area = cv2.contourArea(approx)
            if area < 100: continue 
            
            region = ET.SubElement(regions, "Region", Id=f"{idx+1}", Type="0", Area=f"{area}", NegativeROA="0")
            vertices = ET.SubElement(region, "Vertices")
            for point in approx:
                x, y = point[0]
                ET.SubElement(vertices, "Vertex", X=f"{x}", Y=f"{y}", Z="0")
                
    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="\t")
    xml_path = save_dir / f"{slide_name}.xml"
    with open(xml_path, "w", encoding="utf-8") as f: f.write(xml_str)
    
    # 첫 번째 레이어의 개수만 반환 (로그용)
    first_key = list(contours_dict.keys())[0]
    return len(contours_dict[first_key]['contours'])

def clean_memory():
    gc.collect()
    torch.cuda.empty_cache()

# ============================================================
# [Step 1] 영역 탐색 (Best Strip Selection)
# ============================================================
def find_best_strip(slide, base_mag):
    w, h = slide.dimensions
    ds_search = base_mag / SEARCH_MAG
    best_level = slide.get_best_level_for_downsample(ds_search)
    lw, lh = slide.level_dimensions[best_level]
    
    try:
        img_low = slide.read_region((0, 0), best_level, (lw, lh)).convert("RGB")
    except:
        return None, None

    tissue_mask = get_tissue_mask_robust(img_low) 
    mask_h, mask_w = tissue_mask.shape
    
    sec_w = mask_w // 3
    scores = []
    
    for i in range(3):
        start_x = i * sec_w
        end_x = (i + 1) * sec_w if i < 2 else mask_w
        roi = tissue_mask[:, start_x:end_x]
        score = np.count_nonzero(roi)
        scores.append(score)
        
    best_idx = np.argmax(scores)
    orig_sec_w = w // 3
    crop_x_start = best_idx * orig_sec_w
    crop_x_end = (best_idx + 1) * orig_sec_w if best_idx < 2 else w
    
    print(f"    -> [ROI Selection] Best Strip: {best_idx+1}/3 (Pixel Score: {scores[best_idx]})")
    del img_low, tissue_mask
    return crop_x_start, crop_x_end

# ============================================================
# [Step 2] 통합 Segmentation (Extrarenal 추가됨)
# ============================================================
def run_segmentation_direct(slide, models, x_start, x_end, base_mag):
    w, h = slide.dimensions
    infer_ds = base_mag / SEG_TARGET_MAG 
    
    read_size = int(SEG_PATCH_SIZE * infer_ds) 
    step_size = int(read_size * (1 - SEG_OVERLAP_RATIO))
    
    # 좌표 생성 (썸네일 기반 필터링)
    coords = []
    thumb_ds = 16.0 
    t_level = slide.get_best_level_for_downsample(thumb_ds)
    t_img = slide.read_region((0,0), t_level, slide.level_dimensions[t_level]).convert("RGB")
    t_mask = get_tissue_mask_robust(t_img)
    tm_h, tm_w = t_mask.shape
    tm_scale_x, tm_scale_y = w / tm_w, h / tm_h
    
    for y in range(0, h, step_size):
        for x in range(x_start, x_end, step_size):
            if x + read_size > w or y + read_size > h: continue
            
            tx, ty = int(x / tm_scale_x), int(y / tm_scale_y)
            tw, th = int(read_size / tm_scale_x), int(read_size / tm_scale_y)
            m_roi = t_mask[ty:ty+th, tx:tx+tw]
            
            if m_roi.size > 0 and (np.count_nonzero(m_roi)/m_roi.size) >= 0.1:
                coords.append((x, y))
                
    if not coords:
        return None, None

    strip_w = x_end - x_start
    mask_scale = 1 / infer_ds 
    canvas_w = int(strip_w * mask_scale) + SEG_PATCH_SIZE
    canvas_h = int(h * mask_scale) + SEG_PATCH_SIZE
    
    # Masks
    full_mask_glom = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    full_mask_ifta_1 = np.zeros((canvas_h, canvas_w), dtype=np.uint8) # IFTA
    full_mask_extra_3 = np.zeros((canvas_h, canvas_w), dtype=np.uint8) # Extrarenal (추가됨)
    
    transform = A.Compose([A.Normalize(), ToTensorV2()])
    
    for i in tqdm(range(0, len(coords), BATCH_SIZE), desc="    -> Inference", leave=False):
        batch = coords[i:i+BATCH_SIZE]
        images = []
        valid_indices = []
        
        for idx, (bx, by) in enumerate(batch):
            try:
                img = slide.read_region((bx, by), 0, (read_size, read_size)).convert("RGB")
                img = img.resize((SEG_PATCH_SIZE, SEG_PATCH_SIZE), Image.BICUBIC)
                images.append(np.array(img))
                valid_indices.append(idx)
            except: continue
            
        if not images: continue
        
        batch_tensors = [transform(image=img)["image"] for img in images]
        input_tensor = torch.stack(batch_tensors).to(DEVICE)
        
        with torch.no_grad():
            # Glom
            p_glom = torch.sigmoid(models['glom'](input_tensor))
            p_glom = (p_glom > SEG_THRESH_GLOM).cpu().numpy().astype(np.uint8)
            
            # IFTA (Multi-class)
            logits_ifta = models['ifta'](input_tensor)
            p_ifta = torch.argmax(torch.softmax(logits_ifta, dim=1), dim=1).cpu().numpy().astype(np.uint8)
            
        # Stitching
        for k, v_idx in enumerate(valid_indices):
            gx, gy = batch[v_idx]
            rel_x = gx - x_start
            sx = int(rel_x * mask_scale)
            sy = int(gy * mask_scale)
            ex, ey = sx + SEG_PATCH_SIZE, sy + SEG_PATCH_SIZE
            
            # Glom
            mask_g = p_glom[k, 0]
            if np.sum(mask_g) > (SEG_PATCH_SIZE**2 * GLOM_AREA_RATIO_THRESH):
                full_mask_glom[sy:ey, sx:ex] = cv2.bitwise_or(full_mask_glom[sy:ey, sx:ex], mask_g)
                
            # IFTA (Class 1)
            mask_i = (p_ifta[k] == 1).astype(np.uint8)
            full_mask_ifta_1[sy:ey, sx:ex] = cv2.bitwise_or(full_mask_ifta_1[sy:ey, sx:ex], mask_i)

            # Extrarenal (Class 3) - [추가됨]
            mask_e = (p_ifta[k] == 3).astype(np.uint8)
            full_mask_extra_3[sy:ey, sx:ex] = cv2.bitwise_or(full_mask_extra_3[sy:ey, sx:ex], mask_e)

    def extract_and_restore(mask, class_id, color):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_cnts = []
        for c in cnts:
            c = c * infer_ds
            c[:, 0, 0] += x_start
            final_cnts.append(c.astype(np.int32))
        return {class_id: {"id": "1", "color": color, "contours": final_cnts}}

    # Glom Data
    glom_data = extract_and_restore(full_mask_glom, "Glomerulus", "65280") 
    
    # IFTA Data (Include Extrarenal)
    # IFTA XML에 Extrarenal도 같이 저장 (Visual Check용)
    ifta_contours = extract_and_restore(full_mask_ifta_1, "IFTA", "16711680")['IFTA']['contours']
    extra_contours = extract_and_restore(full_mask_extra_3, "Extrarenal", "16711935")['Extrarenal']['contours']
    
    ifta_data = {
        "IFTA": {"id": "1", "color": "16711680", "contours": ifta_contours},
        "Extrarenal": {"id": "3", "color": "16711935", "contours": extra_contours}
    }
    
    del full_mask_glom, full_mask_ifta_1, full_mask_extra_3
    return glom_data, ifta_data

# ============================================================
# [Step 3] H5 추출 (Discard 로직 추가됨)
# ============================================================
def create_layer_mask_from_data(contours, shape, downsample_ratio):
    mask = np.zeros(shape, dtype=np.uint8)
    scaled_cnts = []
    for cnt in contours:
        c_new = (cnt / downsample_ratio).astype(np.int32)
        scaled_cnts.append(c_new)
    if scaled_cnts:
        cv2.drawContours(mask, scaled_cnts, -1, 255, -1)
    return mask

def run_h5_extraction(slide, slide_name, glom_data, ifta_data, x_start, x_end, base_mag):
    save_path = SAVE_H5_ROOT / f"{slide_name}.h5"
    w, h = slide.dimensions
    ds_factor = base_mag / H5_TARGET_MAG 
    extract_size_l0 = int(H5_PATCH_SIZE * ds_factor) 
    
    mask_w = int(w / ds_factor)
    mask_h = int(h / ds_factor)
    
    print(f"    -> [H5] Mask Generation (Size: {mask_w}x{mask_h})...")
    
    # Masks (20x Scale)
    mask_glom = create_layer_mask_from_data(glom_data['Glomerulus']['contours'], (mask_h, mask_w), ds_factor)
    mask_ifta = create_layer_mask_from_data(ifta_data['IFTA']['contours'], (mask_h, mask_w), ds_factor)
    
    # [추가됨] Extrarenal Mask 생성
    mask_extra = create_layer_mask_from_data(ifta_data['Extrarenal']['contours'], (mask_h, mask_w), ds_factor)
    
    # Tissue Mask (Low Res -> Scaled logic)
    thumb_ds = 32.0 
    th_level = slide.get_best_level_for_downsample(thumb_ds)
    
    # [수정완료] t_level -> th_level 로 변경
    th_img = slide.read_region((0,0), th_level, slide.level_dimensions[th_level]).convert("RGB")
    
    mask_tissue_low = get_tissue_mask_robust(th_img)
    
    mt_h, mt_w = mask_tissue_low.shape
    scale_x_tissue = w / mt_w
    scale_y_tissue = h / mt_h
    
    counts = {"Glomerulus": 0, "IFTA": 0, "Normal": 0, "Discard": 0}
    
    print(f"    -> [H5] Extracting Patches...")
    
    with h5py.File(save_path, 'w') as hf:
        g_glom = hf.create_group("Glomerulus")
        g_ifta = hf.create_group("IFTA")
        g_normal = hf.create_group("Normal")
        
        hf.attrs['slide_id'] = slide_name
        hf.attrs['magnification'] = H5_TARGET_MAG
        
        for y in range(0, h, extract_size_l0):
            for x in range(x_start, x_end, extract_size_l0):
                if x + extract_size_l0 > w or y + extract_size_l0 > h: continue
                
                # 1. Tissue Check
                tx = int(x / scale_x_tissue)
                ty = int(y / scale_y_tissue)
                tw = int(extract_size_l0 / scale_x_tissue)
                th = int(extract_size_l0 / scale_y_tissue)
                
                roi_t = mask_tissue_low[ty:ty+th, tx:tx+tw]
                if roi_t.size == 0: continue
                # Tissue Ratio Check
                if (np.count_nonzero(roi_t) / roi_t.size) < H5_TH_TISSUE: continue
                
                # 2. Mask 좌표 (20x)
                mx = int(x / ds_factor)
                my = int(y / ds_factor)
                mw = H5_PATCH_SIZE 
                mh = H5_PATCH_SIZE
                
                # [추가됨] Extrarenal Discard Logic
                roi_e = mask_extra[my:my+mh, mx:mx+mw]
                if (np.count_nonzero(roi_e) / (mw*mh)) > H5_TH_EXTRA:
                    counts["Discard"] += 1
                    continue # Skip this patch
                
                # 3. Classify
                roi_g = mask_glom[my:my+mh, mx:mx+mw]
                roi_i = mask_ifta[my:my+mh, mx:mx+mw]
                
                label = "Normal"
                if (np.count_nonzero(roi_g) / (mw*mh)) > H5_TH_GLOM:
                    label = "Glomerulus"
                elif (np.count_nonzero(roi_i) / (mw*mh)) > H5_TH_IFTA:
                    label = "IFTA"
                
                # 4. Save
                patch_img = slide.read_region((x, y), 0, (extract_size_l0, extract_size_l0)).convert("RGB")
                if extract_size_l0 != H5_PATCH_SIZE:
                    patch_img = patch_img.resize((H5_PATCH_SIZE, H5_PATCH_SIZE), Image.BICUBIC)
                
                img_np = np.array(patch_img)
                encoded = cv2.imencode(".jpg", cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 90])[1]
                
                key = f"{x}_{y}"
                if label == "Glomerulus": g_glom.create_dataset(key, data=encoded)
                elif label == "IFTA": g_ifta.create_dataset(key, data=encoded)
                else: g_normal.create_dataset(key, data=encoded)
                
                counts[label] += 1
                
    print(f"    -> [H5] 완료: {counts}")
    del mask_glom, mask_ifta, mask_extra, mask_tissue_low
    return counts

# ============================================================
# [Main Pipeline]
# ============================================================
def process_one_slide(slide_name, models):
    start_t = time.time()
    svs_path = SVS_ROOT / f"{slide_name}.svs"
    if not svs_path.exists():
        svs_path = SVS_ROOT / f"{slide_name}_PAS.svs"
        if not svs_path.exists():
            print(f"[Skip] File not found: {slide_name}")
            return

    try:
        slide = openslide.OpenSlide(str(svs_path))
    except Exception as e:
        print(f"[Error] OpenSlide Fail: {e}")
        return

    try: base_mag = float(slide.properties.get("aperio.AppMag", 40.0))
    except: base_mag = 40.0

    # 1. Best Strip
    x_start, x_end = find_best_strip(slide, base_mag)
    if x_start is None:
        print("[Error] ROI 탐색 실패")
        slide.close()
        return

    # 2. Segmentation
    glom_data, ifta_data = run_segmentation_direct(slide, models, x_start, x_end, base_mag)
    if glom_data is None:
        slide.close()
        return

    # 3. XML Save
    try: mpp = float(slide.properties.get("aperio.MPP", 0.2527))
    except: mpp = 0.2527
    
    cnt_g = save_xml(slide_name, glom_data, ANNOTATION_GLOM_ROOT, mpp)
    # IFTA XML에는 Extrarenal 정보도 포함됨
    cnt_i = save_xml(slide_name, ifta_data, ANNOTATION_IFTA_ROOT, mpp)
    print(f"    -> [XML] Saved.")

    # 4. H5 Extraction
    h5_counts = run_h5_extraction(slide, slide_name, glom_data, ifta_data, x_start, x_end, base_mag)

    # 5. Log
    log_data = [slide_name, cnt_g, cnt_i, h5_counts['Glomerulus'], h5_counts['IFTA'], h5_counts['Normal'], h5_counts['Discard'], f"{time.time()-start_t:.1f}"]
    
    write_header = not LOG_CSV_PATH.exists()
    with open(LOG_CSV_PATH, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["ID", "Glom_Cnt", "IFTA_Cnt", "H5_Glom", "H5_IFTA", "H5_Norm", "H5_Discard", "Time_Sec"])
        writer.writerow(log_data)

    slide.close()
    clean_memory()
    print("------------------------------------------------------------")

if __name__ == "__main__":
    df = pd.read_csv(DATASET_CSV)
    df_clean = df[((df['split'] == 'train') & (df['GT'].isnull()))].reset_index(drop=True)
    target_list = df_clean['SlideName'].tolist()
    
    print(f">>> Processing {len(target_list)} slides...")
    models = load_models()
    
    for s_name in target_list:
        print(f"\n>>> [Processing] {s_name}")
        process_one_slide(s_name, models)
        
    print("\n>>> All Process Finished.")
