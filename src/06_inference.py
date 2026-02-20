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
import torch
import pandas as pd
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import math

# 시각화 라이브러리(matplotlib, seaborn)는 사용하지 않으므로 제거함

# ============================================================
# [Settings]
# ============================================================
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# [수정] 모든 경로의 기준이 되는 Root
BASE_DIR = Path("/home/khdp-user/workspace")

# [절대 경로 확인] 아래 경로들은 BASE_DIR을 사용하여 모두 절대 경로임
FEATURE_ROOT = BASE_DIR / "GigaPath/x20_224x224_to_224x224_Stride224v3_Normalized_Merged"
CSV_PATH = BASE_DIR / "dataset/CSV/GT_label.csv"
MODEL_ROOT_DIR = BASE_DIR / "yoongeol/MIL_run_gigapath_compare_CV/type2_TransMIL_AuxLoss_TopK_4Types_Merged"

# [수정] 기존 상대 경로였던 것을 절대 경로로 변경 (파일 위치가 workspace 바로 아래라고 가정)
# 만약 파일이 dataset 폴더 안에 있다면: BASE_DIR / "dataset" / "dataset_with_cluster_public_test.csv" 로 수정 필요
TEST_CLUSTER_CSV = BASE_DIR / "dataset_with_cluster_public_test.csv"

# [수정] 최종 결과 저장 경로 설정
FINAL_SAVE_DIR = BASE_DIR / "Final_submission"
FINAL_SAVE_PATH = FINAL_SAVE_DIR / "infer_pred.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_FOLDS = 5

# ★ [중요] 학습 코드와 하이퍼파라미터 동일하게 맞춤 ★
TOP_K_RATIO = 0.5     
MIN_K = 100
MAX_K = 2048
N_PATCH_CLASSES = 4 

# ★ [요청사항] 목표 양성 비율 12.5% 설정 ★
TARGET_POS_RATIO = 0.125 

# ============================================================
# [Dataset] Inference용
# ============================================================
class GigaPathMILDataset(Dataset):
    def __init__(self, df_slide, feature_root: Path):
        self.feature_root = Path(feature_root)
        self.data_list = []
        for _, row in df_slide.iterrows():
            slide_name = str(row["SlideName"])
            h5_path = self.feature_root / f"{slide_name}.h5"
            if h5_path.exists():
                self.data_list.append({"slide_name": slide_name, "path": h5_path})
                
    def __len__(self): return len(self.data_list)
    
    def __getitem__(self, idx):
        item = self.data_list[idx]
        try:
            with h5py.File(item['path'], 'r') as f:
                feats = []
                target_keys = ['Normal', 'IFTA', 'Glomerulus', 'M0', 'M1']
                
                for cls in target_keys:
                    if cls in f:
                        e = f[cls]['emb'][:]
                        if len(e) > 0: feats.append(e)
                        
                if not feats: return torch.zeros(1,1536), item['slide_name']
                
                features = np.concatenate(feats)
                features_tensor = torch.from_numpy(features).float()
                features_norm = F.layer_norm(features_tensor, (1536,))
                
                return features_norm, item['slide_name']
        except: return torch.zeros(1,1536), item['slide_name']

# ============================================================
# [Model] TransMIL_Aux
# ============================================================
class TransMIL_Aux(nn.Module):
    def __init__(self, input_dim=1536, n_classes=2, n_patch_classes=4, top_k_ratio=0.5, min_k=100, max_k=1024):
        super(TransMIL_Aux, self).__init__()
        self.top_k_ratio = top_k_ratio
        self.min_k = min_k
        self.max_k = max_k
        
        self.pos_layer = PPEG(dim=512)
        self._fc1 = nn.Sequential(nn.Linear(input_dim, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.n_classes = n_classes
        
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        
        self._fc2 = nn.Linear(512, self.n_classes)
        self.aux_classifier = nn.Linear(512, n_patch_classes)

    def forward(self, x):
        patch_scores = torch.norm(x, p=2, dim=-1)
        N = x.shape[1]
        k = int(N * self.top_k_ratio)
        k = max(self.min_k, min(k, self.max_k))
        k = min(k, N)

        topk_indices = None
        if k < N and k > 0:
            _, topk_indices = torch.topk(patch_scores, k, dim=1)
            x = torch.gather(x, 1, topk_indices.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        
        h = self._fc1(x) 
        h = self.pos_layer(h, int(math.sqrt(h.shape[1]))) 
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        h = torch.cat((cls_tokens, h), dim=1)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.norm(h)
        
        slide_logits = self._fc2(h[:, 0])
        patch_logits = self.aux_classifier(h[:, 1:])
        
        return slide_logits, patch_logits, topk_indices

class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)
    def forward(self, x, H): return x 

class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(dim=dim, dim_head=dim//8, heads=8, num_landmarks=256)
    def forward(self, x):
        x = x + self.attn(self.norm(x))
        return x

class NystromAttention(nn.Module):
    def __init__(self, dim, dim_head, heads, num_landmarks=256):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.inner_dim = heads * dim_head
        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(nn.Linear(self.inner_dim, dim), nn.Dropout(0.1))
    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: t.view(b, n, h, -1).transpose(1, 2), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)

def load_weights(model, path):
    state_dict = torch.load(path, map_location=DEVICE)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'): new_state_dict[k[7:]] = v
        else: new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    return model

# ============================================================
# [Main Logic] 
# ============================================================
def run_integrated_pipeline():
    # 1. 데이터 및 모델 로드
    if not CSV_PATH.exists() or not TEST_CLUSTER_CSV.exists(): 
        print(f"Error: CSV file missing.\nCheck: {CSV_PATH}\nCheck: {TEST_CLUSTER_CSV}")
        return

    df = pd.read_csv(CSV_PATH)
    if 'split' in df.columns: df = df[df["split"].str.strip().str.lower() == "test"]
    elif 'GT' in df.columns: df = df[df["GT"].astype(str).str.strip().str.lower() == "test"]
    
    # [주의] Cluster 파일은 Test 셋 리스트를 확보하기 위해 사용 (Threshold 계산용 아님)
    df_cluster = pd.read_csv(TEST_CLUSTER_CSV)
    df_test = pd.merge(df, df_cluster, left_on="SlideName", right_on="SlideName", how="inner").reset_index(drop=True)
    
    print(f"🚀 Total Test Samples: {len(df_test)}")
    
    models = []
    print("\nLoading Models...")
    for fold in range(N_FOLDS):
        path = MODEL_ROOT_DIR / f"best_model_fold{fold}.pt"
        if path.exists():
            m = TransMIL_Aux(n_classes=2, n_patch_classes=N_PATCH_CLASSES, 
                             top_k_ratio=TOP_K_RATIO, min_k=MIN_K, max_k=MAX_K).to(DEVICE)
            load_weights(m, path)
            m.eval()
            models.append(m)
            print(f"Fold {fold} loaded.")
    
    if not models: print("No models loaded"); return

    # 2. Inference
    print("\n[Step 1] Running Inference...")
    ds = GigaPathMILDataset(df_test, FEATURE_ROOT)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)
    raw_results = []
    
    with torch.no_grad():
        for x, s_names in tqdm(loader, desc="Inference"):
            x = x.to(DEVICE)
            if x.dim() == 2: x = x.unsqueeze(0)
            
            slide_probs = []
            for m in models:
                logits, _, _ = m(x) 
                prob = torch.softmax(logits, dim=1)[0, 1].item()
                slide_probs.append(prob)
                
            raw_results.append({"ID": s_names[0], "Prob": np.mean(slide_probs)})
            
    df_final = pd.DataFrame(raw_results)
    
    # --------------------------------------------------------
    # [Step 2] GLOBAL Threshold 계산 (통합 비율)
    # --------------------------------------------------------
    all_probs = df_final["Prob"].values
    target_percentile = (1.0 - TARGET_POS_RATIO) * 100
    global_threshold = np.percentile(all_probs, target_percentile)
    
    df_final["Label_Global"] = (df_final["Prob"] >= global_threshold).astype(int)
    
    print(f"\n{'='*60}")
    print(f" 🌍 GLOBAL Strategy (Target Top {TARGET_POS_RATIO*100}%)")
    print(f"    Global Threshold: {global_threshold:.4f}")
    print(f"    Total Positive: {sum(df_final['Label_Global'])} / {len(df_final)} ({sum(df_final['Label_Global'])/len(df_final)*100:.1f}%)")
    print(f"{'='*60}")

    # --------------------------------------------------------
    # [Step 3] 결과 저장 (Global 기준 단일 파일)
    # --------------------------------------------------------
    # 저장 폴더 생성
    FINAL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    
    # 요청하신 컬럼명 및 형식으로 저장
    # 필요하다면 컬럼명을 변경: Label_Global -> Predicted_Label, Prob -> Predicted_Prob
    sub_global = df_final[["ID", "Label_Global", "Prob"]].rename(
        columns={"Label_Global": "Predicted_Label", "Prob": "Predicted_Prob"}
    )
    
    sub_global.to_csv(FINAL_SAVE_PATH, index=False)
    
    print("\n" + "="*60)
    print(f"✅ Finished! (Ratio: {TARGET_POS_RATIO*100}%)")
    print(f"📂 Saved to: {FINAL_SAVE_PATH}")
    print("="*60)

if __name__ == "__main__":
    run_integrated_pipeline()
