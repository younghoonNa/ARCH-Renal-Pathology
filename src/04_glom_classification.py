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
import torchvision.transforms as T
import timm
from torch.utils.data import Dataset, DataLoader
import torchstain

# ============================================================
# [설정]
# ============================================================
os.environ["HF_HUB_DISABLE_XET"] = "1"

# 원본 (이미지 바이트 + Predictions/threshold 있는 H5들)
SOURCE_DIR = Path("/home/khdp-user/workspace/Patch_3_Glom_final")

# 최종 출력 (M0/M1 분리된 결과만 저장)
NEW_OUT_DIR = Path("/home/khdp-user/workspace/GigaPath/x20_224x224_to_224x224_Stride224v3_Normalized_M0M1C")
NEW_OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "hf_hub:prov-gigapath/prov-gigapath"
CLASSES = ["Glomerulus"]

BATCH_SIZE = 512
NUM_WORKERS = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 키에서 g index 추출 (예: xxx_g123)
KEY_RE = re.compile(r".*_g(\d+)$")


# ============================================================
# [Macenko Normalizer] - 수치 고정 (파일 불필요)
# ============================================================
class MacenkoNormalizer:
    def __init__(self):
        self.n = torchstain.normalizers.MacenkoNormalizer(backend="torch")

        self.n.HERef = torch.tensor(
            [
                [0.5626, 0.2159],
                [0.7201, 0.8012],
                [0.4062, 0.5581],
            ],
            device=DEVICE,
        )
        self.n.maxCRef = torch.tensor([1.9705, 1.0308], device=DEVICE)

    def __call__(self, img_np: np.ndarray) -> Image.Image:
        # 배경(조직 없음) 스킵
        if np.count_nonzero(np.mean(img_np, axis=2) < 220) < 100:
            return Image.fromarray(img_np)

        try:
            I = torch.from_numpy(img_np).to(DEVICE)
            norm, _, _ = self.n.normalize(I=I, stains=False)
            return Image.fromarray(norm.detach().cpu().numpy().astype(np.uint8))
        except Exception:
            return Image.fromarray(img_np)


normalizer = MacenkoNormalizer()

TRANSFORM = T.Compose(
    [
        T.Lambda(lambda x: normalizer(x)),  # Macenko
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# ============================================================
# [모델 로드]
# ============================================================
def load_model():
    print(f"[Info] Loading Model: {MODEL_NAME} on {DEVICE}")
    model = timm.create_model(MODEL_NAME, pretrained=True)
    model.eval().to(DEVICE)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    return model


@torch.inference_mode()
def infer_batch(model, x: torch.Tensor) -> np.ndarray:
    out = model(x)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out.detach().float().cpu().numpy()


# ============================================================
# Dataset (원본 H5에서 바로 패치 로드)
# ============================================================
class PatchH5Dataset(Dataset):
    """
    - 원본 H5의 'Glomerulus' 그룹에 있는 패치들을 읽어서
    - (tensor, idx) 반환
    """

    def __init__(self, h5_path: Path, cls_name: str):
        self.h5_path = h5_path
        self.cls_name = cls_name

        self.hf = h5py.File(h5_path, "r")
        self.group = self.hf.get(cls_name)

        self.keys = []
        self.idxs = []

        if self.group is not None:
            for k in self.group.keys():
                m = KEY_RE.match(k)
                if m:
                    self.keys.append(k)
                    self.idxs.append(int(m.group(1)))

        # 안정적 순서 보장 (idx 기준 정렬)
        if len(self.keys) > 0:
            order = np.argsort(np.array(self.idxs))
            self.keys = [self.keys[i] for i in order]
            self.idxs = [self.idxs[i] for i in order]

    def __len__(self):
        return 0 if self.group is None else len(self.keys)

    def __getitem__(self, idx):
        k = self.keys[idx]
        gidx = self.idxs[idx]

        buf = self.group[k][()]
        img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            # 안전장치
            return torch.zeros(3, 224, 224), np.int32(gidx)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = TRANSFORM(img_rgb)
        return x, np.int32(gidx)

    def close(self):
        try:
            self.hf.close()
        except Exception:
            pass

    def __del__(self):
        self.close()


# ============================================================
# [원본 H5에서 threshold + prob 읽기]
# ============================================================
def load_probs_and_threshold(src_path: Path):
    with h5py.File(src_path, "r") as f:
        if "Predictions" not in f:
            return None, None, "No Predictions in source"

        pred_grp = f["Predictions"]
        if "threshold" not in pred_grp.attrs:
            return None, None, "No threshold attribute in source"

        threshold = float(pred_grp.attrs["threshold"])

        if "Glomerulus" not in f:
            return None, None, "No Glomerulus group in source"

        idx_to_prob = {}
        for k in f["Glomerulus"].keys():
            m = KEY_RE.match(k)
            if not m:
                continue
            gidx = int(m.group(1))
            if k in pred_grp:
                idx_to_prob[gidx] = float(pred_grp[k][()])
            else:
                idx_to_prob[gidx] = -1.0

    return idx_to_prob, threshold, None


# ============================================================
# [슬라이드 1개: 임베딩 생성 + 바로 M0/M1 split + 저장]
# ============================================================
def process_one_slide(model, src_path: Path, out_path: Path):
    slide_id = src_path.stem

    # 1) probs/threshold 로드
    idx_to_prob, threshold, err = load_probs_and_threshold(src_path)
    if err is not None:
        return False, err

    # 2) dataset/loader
    ds = PatchH5Dataset(src_path, "Glomerulus")
    if len(ds) == 0:
        ds.close()
        return False, "No patches in Glomerulus"

    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # 3) 임베딩+확률 정렬 수집
    embs_all = []
    coords_all = []
    probs_all = []

    for imgs, gidxs in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        emb = infer_batch(model, imgs)  # (B, 1536)

        gidxs_np = gidxs.numpy().astype(np.int32)  # (B,)
        # coords: (idx, 0) 형태로 저장
        coords = np.stack([gidxs_np, np.zeros_like(gidxs_np)], axis=1).astype(np.int32)

        # probs aligned
        # 매칭 실패는 -1.0
        probs = np.array([idx_to_prob.get(int(i), -1.0) for i in gidxs_np], dtype=np.float32)

        embs_all.append(emb.astype(np.float16))
        coords_all.append(coords)
        probs_all.append(probs.astype(np.float16))

    ds.close()

    embeddings = np.concatenate(embs_all, axis=0)  # float16
    coords = np.concatenate(coords_all, axis=0)    # int32
    probs = np.concatenate(probs_all, axis=0)      # float16 (but values are probs)

    # 4) split
    probs_f32 = probs.astype(np.float32)
    valid_mask = probs_f32 >= 0.0
    if valid_mask.sum() == 0:
        return False, "No valid probabilities matched"

    mask_m1 = (probs_f32 > threshold) & valid_mask
    mask_m0 = (probs_f32 <= threshold) & valid_mask

    emb_m1 = embeddings[mask_m1].astype(np.float16)
    coord_m1 = coords[mask_m1].astype(np.int32)
    prob_m1 = probs[mask_m1].astype(np.float16)

    emb_m0 = embeddings[mask_m0].astype(np.float16)
    coord_m0 = coords[mask_m0].astype(np.int32)
    prob_m0 = probs[mask_m0].astype(np.float16)

    # 5) 저장 (최종 파일만 생성)
    with h5py.File(out_path, "w") as f_out:
        # 원본 attrs 복사
        with h5py.File(src_path, "r") as f_src:
            for k, v in f_src.attrs.items():
                try:
                    f_out.attrs[k] = v
                except Exception:
                    pass

        # 메타데이터
        f_out.attrs["slide_id"] = slide_id
        f_out.attrs["model"] = MODEL_NAME
        f_out.attrs["embed_dim"] = 1536
        f_out.attrs["patch_size"] = 224
        f_out.attrs["stride"] = 224
        f_out.attrs["magnification"] = 20
        f_out.attrs["normalization"] = "Macenko_Hardcoded"
        f_out.attrs["threshold_used"] = float(threshold)
        f_out.attrs["split_type"] = "M0_M1_separated_v2"

        if emb_m1.shape[0] > 0:
            g1 = f_out.create_group("M1")
            g1.create_dataset("emb", data=emb_m1, compression="gzip", compression_opts=4)
            g1.create_dataset("coords", data=coord_m1, compression="gzip", compression_opts=4)
            g1.create_dataset("probs", data=prob_m1, compression="gzip", compression_opts=4)
            g1.attrs["n_patches"] = int(emb_m1.shape[0])

        if emb_m0.shape[0] > 0:
            g0 = f_out.create_group("M0")
            g0.create_dataset("emb", data=emb_m0, compression="gzip", compression_opts=4)
            g0.create_dataset("coords", data=coord_m0, compression="gzip", compression_opts=4)
            g0.create_dataset("probs", data=prob_m0, compression="gzip", compression_opts=4)
            g0.attrs["n_patches"] = int(emb_m0.shape[0])

    msg = f"OK (thr={threshold:.4f}) -> M0={emb_m0.shape[0]}, M1={emb_m1.shape[0]}"
    return True, msg


# ============================================================
# [Main]
# ============================================================
def main():
    torch.backends.cudnn.benchmark = True

    model = load_model()

    src_files = sorted(SOURCE_DIR.glob("*.h5"))
    print(f"[Start] One-pass embed + split")
    print(f"Source: {SOURCE_DIR}")
    print(f"Output: {NEW_OUT_DIR}")
    print(f"Files : {len(src_files)}")

    success = 0
    failed = 0

    for src_path in tqdm(src_files, desc="Slides"):
        out_path = NEW_OUT_DIR / src_path.name

        try:
            ok, msg = process_one_slide(model, src_path, out_path)
            if ok:
                success += 1
            else:
                failed += 1
                print(f"[Fail] {src_path.name}: {msg}")
        except Exception as e:
            failed += 1
            print(f"[Error] {src_path.name}: {e}")

    print(f"\n[Done] Success={success}, Failed={failed}")
    print(f"Output dir: {NEW_OUT_DIR}")


if __name__ == "__main__":
    main()
