import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler   # hoặc MinMaxScaler

# >>> CopulaGAN (SDV) <<<
from sdv.single_table import CopulaGANSynthesizer
from sdv.metadata import SingleTableMetadata

# ------------------------------------------------------------
# 1. Load & chuẩn bị dữ liệu
# ------------------------------------------------------------

df_ls    = pd.read_excel("LS_2_aftercheck.xlsx")       # Landslide points
df_nonls = pd.read_excel("Non_LS_2_aftercheck.xlsx")   # Non-landslide points

# Gán nhãn
df_ls["Label"]    = 1
df_nonls["Label"] = 0

label_col   = "Label"
random_state = 10

# ------------------------------------------------------------
# 2. Chia Landslide: df_ls -> ls_train & ls_test
# ------------------------------------------------------------

ls_train, ls_test = train_test_split(
    df_ls,
    test_size=0.3,
    shuffle=True,
    random_state=random_state
)

print("LS total     :", df_ls.shape)
print("LS train     :", ls_train.shape)
print("LS test      :", ls_test.shape)

# ------------------------------------------------------------
# 3. Chia tiếp ls_train thành 2 tập nhỏ (A/B)
# ------------------------------------------------------------

ls_train_A, ls_train_B = train_test_split(
    ls_train,
    test_size=0.001,   # gần như dùng hết ls_train cho train_A
    shuffle=True,
    random_state=random_state
)

print("LS train_A   :", ls_train_A.shape)
print("LS train_B   :", ls_train_B.shape, "(có thể dùng làm validation riêng)")

# ------------------------------------------------------------
# 4. Chia Non-landslide: df_nonls -> nonls_train & nonls_test
# ------------------------------------------------------------

nonls_train, nonls_test = train_test_split(
    df_nonls,
    test_size=0.3,
    shuffle=True,
    random_state=random_state
)

print("Non-LS total :", df_nonls.shape)
print("Non-LS train :", nonls_train.shape)
print("Non-LS test  :", nonls_test.shape)

# Chia tiếp nonls_train thành 2 phần A/B
nonls_train_A, nonls_train_B = train_test_split(
    nonls_train,
    test_size=0.001,
    shuffle=True,
    random_state=random_state
)

print("nonls train_A:", nonls_train_A.shape)
print("nonls train_B:", nonls_train_B.shape, "(có thể dùng làm validation riêng)")

# ------------------------------------------------------------
# 5. Ghép lại tập TRAIN & TEST cho mô hình (trước khi thêm CopulaGAN)
# ------------------------------------------------------------

train_df = pd.concat([ls_train_A, nonls_train_A], ignore_index=True)
test_df  = pd.concat([ls_test,    nonls_test],    ignore_index=True)

train_df = train_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
test_df  = test_df.sample(frac=1.0,  random_state=random_state).reset_index(drop=True)

print("\n==> Tập cuối cùng ban đầu:")
print("TRAIN total  :", train_df.shape)
print("TEST total   :", test_df.shape)

val_df = ls_train_B.copy()  # nếu cần dùng validation riêng cho LS

# ------------------------------------------------------------
# 6. CopulaGAN: sinh thêm 100 landslide (Label = 1)
# ------------------------------------------------------------

print("\nColumns in train_df:", list(train_df.columns))

# Lọc landslide trong train_df để train CopulaGAN
ls_train_for_gan = train_df[train_df[label_col] == 1].copy()
print("\nLS rows used to train CopulaGAN:", ls_train_for_gan.shape)

# Tạo metadata
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(ls_train_for_gan)

# Khai báo các cột categorical (sửa tên cho đúng với Excel của bạn)
categorical_cols = [
    "Soil_type_",   # nhớ khớp tên column trong file
    "Geology_nu",
    "Forest_den",
    "Timper_age",
    "Diameter",
    "Forest_typ",
    "Label"
]

for col in categorical_cols:
    if col in ls_train_for_gan.columns:
        metadata.update_column(col, sdtype="categorical")

# Khởi tạo & train CopulaGAN
copula_gan = CopulaGANSynthesizer(
    metadata=metadata,
    epochs=300,      # bạn có thể tăng/giảm
    verbose=True
)

print("\n=== Training CopulaGAN on LS data ===")
copula_gan.fit(ls_train_for_gan)

# Sample 100 LS synthetic
synthetic_ls_copula = copula_gan.sample(100)
synthetic_ls_copula[label_col] = 1   # đảm bảo nhãn 1

print("\n========== 100 LANDSLIDE SYNTHETIC (CopulaGAN) ==========")
print(synthetic_ls_copula.head())

# Lưu ra Excel
synthetic_ls_copula.to_excel("LS_2_aftercheck_CopulaGAN_100.xlsx", index=False)
print("\nSaved 100 new LS samples (CopulaGAN) to: LS_2_aftercheck_CopulaGAN_100.xlsx")

# ------------------------------------------------------------
# 6.x Gộp 100 LS synthetic vào train_df
# ------------------------------------------------------------

train_df_aug = pd.concat([train_df, synthetic_ls_copula], ignore_index=True)
train_df_aug = train_df_aug.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

print("\n==> Sau CopulaGAN augmentation:")
print("TRAIN total (with synthetic):", train_df_aug.shape)

# ------------------------------------------------------------
# 7. Tạo X_train, y_train, X_test, y_test + Normalize
# ------------------------------------------------------------

X_train_df = train_df_aug.drop(columns=[label_col])
y_train    = train_df_aug[label_col].astype(int)

X_test_df  = test_df.drop(columns=[label_col])
y_test     = test_df[label_col].astype(int)

print("\nFinal feature columns:", list(X_train_df.columns))
feature_names = X_train_df.columns.tolist()
print("Feature count:", len(feature_names))

X_train = X_train_df.values
X_test  = X_test_df.values

# Chuẩn hoá (sau khi đã augment)
scaler = StandardScaler()   # hoặc MinMaxScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)

print("\nNormalize done. Ready for SSA / XGBoost / GWO / WOA ...")
