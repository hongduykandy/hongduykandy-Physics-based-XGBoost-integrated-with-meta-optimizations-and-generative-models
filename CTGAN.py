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

# CTGAN (SDV)
from sdv.single_table import CTGANSynthesizer
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
print("LS train_B   :", ls_train_B.shape, "(validation nếu cần)")

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
print("nonls train_B:", nonls_train_B.shape, "(validation nếu cần)")

# ------------------------------------------------------------
# 5. Ghép lại tập TRAIN & TEST cho mô hình (chưa GAN)
# ------------------------------------------------------------

train_df = pd.concat([ls_train_A, nonls_train_A], ignore_index=True)
test_df  = pd.concat([ls_test,   nonls_test],     ignore_index=True)

train_df = train_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
test_df  = test_df.sample(frac=1.0,  random_state=random_state).reset_index(drop=True)

print("\n==> Tập cuối cùng (trước CTGAN):")
print("TRAIN total  :", train_df.shape)
print("TEST total   :", test_df.shape)

val_df = ls_train_B.copy()  # nếu cần dùng validation riêng cho LS

# ------------------------------------------------------------
# 6. Dùng CTGAN sinh thêm 100 landslide (Label = 1)
# ------------------------------------------------------------

# Lọc landslide trong train_df để train GAN
ls_train_for_gan = train_df[train_df[label_col] == 1].copy()

print("\nLS rows used to train CTGAN:", ls_train_for_gan.shape)

# Tạo metadata cho CTGAN
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(ls_train_for_gan)

# Cập nhật sdtype cho các cột categorical (sửa nếu tên khác trong Excel)
categorical_cols = [
    "Soil_type_",   # khớp với tên cột thật
    "Geology_nu",
    "Forest_den",
    "Timper_age",
    "Diameter",
    "Forest_typ"
]

for col in categorical_cols:
    if col in ls_train_for_gan.columns:
        metadata.update_column(col, sdtype="categorical")

# Label luôn = 1, coi như categorical/binary
metadata.update_column(label_col, sdtype="categorical")

# Khởi tạo & train CTGAN
ctgan = CTGANSynthesizer(
    metadata=metadata,
    epochs=300,
    verbose=True
)

ctgan.fit(ls_train_for_gan)

# Sample 100 landslide synthetic
synthetic_ls_100 = ctgan.sample(100)
synthetic_ls_100[label_col] = 1  # đảm bảo Label = 1

print("\n========== 100 LANDSLIDE SYNTHETIC (CTGAN) ==========")
print(synthetic_ls_100)

# Lưu 100 LS mới ra Excel
synthetic_ls_100.to_excel("LS_2_aftercheck_CTGAN_100.xlsx", index=False)
print("\nSaved 100 new LS (CTGAN) to: LS_2_aftercheck_CTGAN_100.xlsx")

# ------------------------------------------------------------
# 6.1 Gộp 100 LS synthetic vào train_df
# ------------------------------------------------------------

train_df_ctgan = pd.concat([train_df, synthetic_ls_100], ignore_index=True)
train_df_ctgan = train_df_ctgan.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

print("\n==> Sau CTGAN augmentation:")
print("TRAIN total  (with synthetic):", train_df_ctgan.shape)

# ------------------------------------------------------------
# 7. Tạo X_train, y_train, X_test, y_test + Normalize
# ------------------------------------------------------------

X_train_df = train_df_ctgan.drop(columns=[label_col])
y_train    = train_df_ctgan[label_col].astype(int)

X_test_df  = test_df.drop(columns=[label_col])
y_test     = test_df[label_col].astype(int)

feature_names = X_train_df.columns.tolist()
print("\nFeature count:", len(feature_names))

X_train = X_train_df.values
X_test  = X_test_df.values

# Chuẩn hoá
scaler = StandardScaler()       # hoặc MinMaxScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)

print("Normalize done. Ready for SSA + XGBoost.")
