# -*- coding: utf-8 -*-
"""    02_glom_generation.py

목적:
  - 01_segmentation.py 결과물(Patch_3_final/*.h5)에서 Glomerulus 패치만 추출하여
    Patch_3_Glom_final/*.h5 로 저장합니다.

주의:
  - 04_glom_classification.py 는 Glomerulus 그룹 내 dataset key가 '*_g<idx>' 형식이기를 기대합니다.
    그래서 여기서 key를 '<원본키>_g{idx}'로 재부여합니다.
  - 또한 04_glom_classification.py 는 Predictions 그룹과 threshold attribute를 기대합니다.
    원본에 Predictions가 없다면, 이 스크립트는 "placeholder"로 probs=0.0, threshold=0.5 를 생성합니다.
    (실제 M0/M1 확률/threshold 로직이 있다면 해당 부분을 교체하세요.)

HuggingFace 토큰(HF_TOKEN)은 하드코딩하지 않았습니다. 필요 시 환경변수로 주입하세요.
"""

import os
from pathlib import Path
import h5py
from tqdm import tqdm

os.environ["HF_HUB_DISABLE_XET"] = os.environ.get("HF_HUB_DISABLE_XET", "1")

BASE_DIR = Path("/home/khdp-user/workspace")
SRC_DIR = BASE_DIR / "Patch_3_final"
OUT_DIR = BASE_DIR / "Patch_3_Glom_final"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_THRESHOLD = 0.5
DEFAULT_PROB = 0.0

def convert_one(src_path: Path, out_path: Path) -> None:
    with h5py.File(src_path, "r") as f_in, h5py.File(out_path, "w") as f_out:
        # copy attrs
        for k, v in f_in.attrs.items():
            f_out.attrs[k] = v

        # copy Glomerulus group only
        if "Glomerulus" not in f_in:
            return

        g_in = f_in["Glomerulus"]
        g_out = f_out.create_group("Glomerulus")

        # Predictions group (copy if exists, else placeholder)
        pred_out = f_out.create_group("Predictions")

        if "Predictions" in f_in:
            pred_in = f_in["Predictions"]
            # threshold attr copy if exists
            if "threshold" in pred_in.attrs:
                pred_out.attrs["threshold"] = pred_in.attrs["threshold"]
            else:
                pred_out.attrs["threshold"] = DEFAULT_THRESHOLD
        else:
            pred_out.attrs["threshold"] = DEFAULT_THRESHOLD

        # re-key datasets
        for idx, key in enumerate(g_in.keys()):
            new_key = f"{key}_g{idx}"
            g_out.create_dataset(new_key, data=g_in[key][()])

            # copy prob if possible; else placeholder
            prob = DEFAULT_PROB
            if "Predictions" in f_in:
                pred_in = f_in["Predictions"]
                # try to match original key first
                if key in pred_in:
                    try:
                        prob = float(pred_in[key][()])
                    except Exception:
                        prob = DEFAULT_PROB
                elif new_key in pred_in:
                    try:
                        prob = float(pred_in[new_key][()])
                    except Exception:
                        prob = DEFAULT_PROB
            pred_out.create_dataset(new_key, data=prob)

def main():
    h5_paths = sorted(SRC_DIR.glob("*.h5"))
    if not h5_paths:
        print(f"[WARN] No h5 files in {SRC_DIR}")
        return

    print(f">>> Converting {len(h5_paths)} files:")
    for p in tqdm(h5_paths):
        out_path = OUT_DIR / p.name
        convert_one(p, out_path)

    print(f">>> Done. Output: {OUT_DIR}")

if __name__ == "__main__":
    main()
