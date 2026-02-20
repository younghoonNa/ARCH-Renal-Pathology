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
import re
from pathlib import Path
import numpy as np
import h5py
from tqdm import tqdm
import cv2
from PIL import Image

import torch
import torchvision
import timm
from torch.utils.data import Dataset, DataLoader

# Stain Normalization
import torchstain
import torchvision.transforms as T

# ============================================================
# [환경/설정]
# ============================================================
os.environ["HF_HUB_DISABLE_XET"] = os.environ.get("HF_HUB_DISABLE_XET", "1")

PATCH_H5_ROOT = Path("/home/khdp-user/workspace/Patch_3_final")
OUT_DIR = Path("/home/khdp-user/workspace/GigaPath/x20_224x224_to_224x224_Stride224v3_Normalized") 
OUT_DIR.mkdir(parents=True, exist_ok=True)

# [삭제됨] TARGET_IMG_PATH 변수는 더 이상 필요 없습니다.

CLASSES = ["Glomerulus", "IFTA", "Normal"]
MODEL_NAME = "hf_hub:prov-gigapath/prov-gigapath"

BATCH_SIZE = 512
NUM_WORKERS = 8 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
KEY_RE = re.compile(r"^(-?\d+)_(-?\d+)$")

# ============================================================
# [Macenko Normalizer] - 타겟 파일 없이 표준값 고정 (Clean Version)
# ============================================================
class MacenkoNormalizer:
    def __init__(self):
        # backend='torch'로 GPU 가속 활용 가능
        self.n = torchstain.normalizers.MacenkoNormalizer(backend='torch')
        
        # ★ [핵심] 이상적인 H&E 표준 색상 벡터 주입 (파일 불필요)
        # 이 값은 가장 보편적인 H&E 염색의 표준값입니다.
        self.n.HERef = torch.tensor([
            [0.5626, 0.2159],
            [0.7201, 0.8012],
            [0.4062, 0.5581]
        ]).to(DEVICE)
        self.n.maxCRef = torch.tensor([1.9705, 1.0308]).to(DEVICE)

    def __call__(self, img_np):
        """
        img_np: RGB Numpy array
        Returns: PIL Image
        """
        # 1. 흰색 배경(조직 없는 패치) 체크 -> MKL 에러 방지
        # (평균 밝기가 220 미만인 '조직 픽셀'이 100개도 안 되면 정규화 스킵)
        if np.count_nonzero(np.mean(img_np, axis=2) < 220) < 100:
            return Image.fromarray(img_np)

        try:
            # 2. 정규화 수행 (GPU로 보내서 연산)
            I = torch.from_numpy(img_np).to(DEVICE)
            norm, _, _ = self.n.normalize(I=I, stains=False)
            
            # 결과 반환 (CPU -> Numpy -> PIL)
            return Image.fromarray(norm.cpu().numpy().astype(np.uint8))
        except:
            # 계산 실패 시 원본 반환
            return Image.fromarray(img_np)

# ============================================================
# [Transform 구성]
# ============================================================
# 1. Normalizer 초기화 (인자 없음)
macenko_norm = MacenkoNormalizer()

# 2. Transform 파이프라인
TRANSFORM = T.Compose([
    # (Resize는 이미 H5 저장 단계나 데이터셋 로드 후 크기가 맞다고 가정, 필요시 추가)
    # T.Resize(224), 
    
    T.Lambda(lambda x: macenko_norm(x)), # [핵심] Hard-coded Macenko 적용
    
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


# ============================================================
# [모델 로드]
# ============================================================
def load_gigapath():
    print(f"[Info] Loading GigaPath model on {DEVICE} ...")
    model = timm.create_model(MODEL_NAME, pretrained=True)
    model.eval().to(DEVICE)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    return model


# ============================================================
# Dataset
# ============================================================
class PatchH5Dataset(Dataset):
    def __init__(self, h5_path: Path, cls_name: str):
        self.h5_path = h5_path
        self.cls_name = cls_name

        self.hf = h5py.File(h5_path, "r")
        self.group = self.hf[cls_name]

        self.keys = []
        self.coords = []

        for k in self.group.keys():
            m = KEY_RE.match(k)
            if m:
                self.keys.append(k)
                self.coords.append((int(m.group(1)), int(m.group(2))))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        k = self.keys[idx]
        coord = self.coords[idx]

        # 1. H5에서 바이트 읽기
        buf = np.asarray(self.group[k][()], dtype=np.uint8)
        
        # 2. 디코딩 (BGR)
        img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("cv2.imdecode failed")

        # 3. BGR -> RGB 변환 (Numpy)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # 4. Transform 적용 (Macenko -> ToTensor -> Normalize)
        tensor = TRANSFORM(img_rgb)

        return tensor, np.array(coord, dtype=np.int32)

    def close(self):
        self.hf.close()


# ============================================================
# [배치 추론]
# ============================================================
@torch.inference_mode()
def infer_batch(model, x):
    out = model(x)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out.detach().float().cpu().numpy()


# ============================================================
# [슬라이드 하나 처리]
# ============================================================
def process_one_slide_h5(model, in_h5_path: Path, out_h5_path: Path):
    slide_id = in_h5_path.stem
    class_to_result = {}

    with h5py.File(in_h5_path, "r") as hf_in:
        in_attrs = dict(hf_in.attrs)

    for cls in CLASSES:
        with h5py.File(in_h5_path, "r") as hf:
            if cls not in hf:
                continue
            if len(hf[cls].keys()) == 0:
                continue

        dataset = PatchH5Dataset(in_h5_path, cls)
        if len(dataset) == 0:
            dataset.close()
            continue

        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True
        )

        embs_list = []
        coords_list = []

        for imgs, coords in loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            out = infer_batch(model, imgs)
            embs_list.append(out.astype(np.float16))
            coords_list.append(coords.numpy().astype(np.int32))

        dataset.close()

        if len(embs_list) == 0:
            continue

        emb = np.concatenate(embs_list, axis=0)
        crd = np.concatenate(coords_list, axis=0)
        class_to_result[cls] = (emb, crd)

    if len(class_to_result) == 0:
        return False

    # ============================================================
    # 저장
    # ============================================================
    with h5py.File(out_h5_path, "w") as hf_out:
        hf_out.attrs["slide_id"] = slide_id
        hf_out.attrs["model"] = MODEL_NAME
        hf_out.attrs["embed_dim"] = 1536
        hf_out.attrs["patch_size"] = 224
        hf_out.attrs["stride"] = 224
        hf_out.attrs["magnification"] = 20
        hf_out.attrs["normalization"] = "Macenko_Hardcoded" # 메타데이터 변경

        for k, v in in_attrs.items():
            if k not in hf_out.attrs:
                try:
                    hf_out.attrs[k] = v
                except:
                    pass

        for cls in CLASSES:
            g = hf_out.create_group(cls)
            if cls in class_to_result:
                emb, crd = class_to_result[cls]
            else:
                emb = np.zeros((0, 1536), dtype=np.float16)
                crd = np.zeros((0, 2), dtype=np.int32)

            g.create_dataset("emb", data=emb, compression="gzip", compression_opts=4)
            g.create_dataset("coords", data=crd, compression="gzip", compression_opts=4)
            g.attrs["n_patches"] = int(emb.shape[0])

    return True


# ============================================================
# [전체 실행]
# ============================================================
def main():
    torch.backends.cudnn.benchmark = True
    
    # [삭제됨] 타겟 이미지 존재 확인 로직 제거

    model = load_gigapath()

    in_list = sorted(PATCH_H5_ROOT.glob("*.h5"))

    for e in in_list:
        if str(e).split("/")[-1][:-3] in TARGET_LIST:
            print(e)
    print(f"[Start] Found {len(in_list)} patch-h5 files")
    print(f"[Info] Normalization: Standard Macenko (Hard-coded Reference)")

    saved = 0
    skipped = 0

    for in_h5 in tqdm(in_list, desc="Slides"):    
        out_h5 = OUT_DIR / in_h5.name
        try:
            ok = process_one_slide_h5(model, in_h5, out_h5)
            if ok:
                saved += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"[Error] {in_h5.name} failed: {e}")
            skipped += 1

    print(f"\n[Done] saved={saved}, skipped={skipped}")
    print(f"Output dir: {OUT_DIR}")


if __name__ == "__main__":
    main()
