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

import h5py
import os
import shutil
from pathlib import Path
from tqdm import tqdm

# ============================================================
# [설정] 경로 지정
# ============================================================
# 1. 분류된 M0, M1이 있는 폴더 (Source A)
DIR_M0M1 = Path('/home/khdp-user/workspace/GigaPath/x20_224x224_to_224x224_Stride224v3_Normalized_M0M1C')

# 2. 전체 조직(Glomerulus, IFTA, Normal)이 있는 폴더 (Source B)
DIR_ORG = Path('/home/khdp-user/workspace/GigaPath/x20_224x224_to_224x224_Stride224v3_Normalized')

# 3. [최종] 합쳐진 파일이 저장될 폴더
DIR_MERGED = Path('/home/khdp-user/workspace/GigaPath/x20_224x224_to_224x224_Stride224v3_Normalized_Merged')
DIR_MERGED.mkdir(parents=True, exist_ok=True)

def merge_h5_files(path_m0m1, path_org, path_out):
    """
    두 H5 파일의 그룹들을 하나로 합칩니다.
    """
    with h5py.File(path_out, 'w') as f_out:
        
        # -------------------------------------------------
        # 1. M0M1 파일에서 M0, M1 가져오기
        # -------------------------------------------------
        with h5py.File(path_m0m1, 'r') as f_src1:
            # 메타데이터(Attributes) 복사 (우선순위 높음)
            for k, v in f_src1.attrs.items():
                f_out.attrs[k] = v
            
            # 그룹 복사 (M0, M1)
            for group_name in ['M0', 'M1']:
                if group_name in f_src1:
                    # copy: 데이터 로드 없이 내부적으로 고속 복사
                    f_src1.copy(group_name, f_out)

        # -------------------------------------------------
        # 2. 원본 파일에서 Glomerulus, IFTA, Normal 가져오기
        # -------------------------------------------------
        with h5py.File(path_org, 'r') as f_src2:
            # 메타데이터 보완 (없으면 추가)
            for k, v in f_src2.attrs.items():
                if k not in f_out.attrs:
                    f_out.attrs[k] = v
            
            # 그룹 복사 (Glomerulus, IFTA, Normal)
            # ★ 사용자의 요청대로 Glomerulus도 그대로 가져옴 (M0/M1과 별개)
            targets = ['Glomerulus', 'IFTA', 'Normal']
            
            for group_name in targets:
                if group_name in f_src2:
                    f_src2.copy(group_name, f_out)

    return True

def main():
    # 파일 목록 기준은 M0M1 폴더 (분류가 완료된 파일들)
    files = sorted(list(DIR_M0M1.glob('*.h5')))
    
    print(f"[Start] Merging H5 files...")
    print(f"Source A (M0/M1): {DIR_M0M1}")
    print(f"Source B (Org):   {DIR_ORG}")
    print(f"Output:           {DIR_MERGED}")
    
    success = 0
    skipped = 0
    
    for f_m0m1 in tqdm(files):
        filename = f_m0m1.name
        f_org = DIR_ORG / filename
        f_out = DIR_MERGED / filename
        
        # 원본(Org) 파일이 존재하는지 확인
        if not f_org.exists():
            # M0M1은 있는데 원본 폴더에 파일이 없는 경우 (드문 케이스)
            print(f"[Skip] Missing original file: {filename}")
            skipped += 1
            continue
            
        try:
            merge_h5_files(f_m0m1, f_org, f_out)
            success += 1
        except Exception as e:
            print(f"[Error] Failed to merge {filename}: {e}")
            skipped += 1
            
    print(f"\n[Done] Successfully merged {success} files.")
    print(f"Files saved to: {DIR_MERGED}")

    # -------------------------------------------------
    # [검증] 첫 번째 파일 구조 출력
    # -------------------------------------------------
    if success > 0:
        print("\n🔎 Verifying the first merged file...")
        first_file = list(DIR_MERGED.glob('*.h5'))[0]
        
        with h5py.File(first_file, 'r') as f:
            print(f"File: {first_file.name}")
            print(f"Groups: {list(f.keys())}")  # ['Glomerulus', 'IFTA', 'M0', 'M1', 'Normal'] 예상
            for k in f.keys():
                # 각 그룹의 shape 확인
                if 'emb' in f[k]:
                    print(f" - {k}: {f[k]['emb'].shape}")

if __name__ == "__main__":
    main()
