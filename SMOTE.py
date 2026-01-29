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

from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import StandardScaler   # hoặc MinMaxScaler

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
# 5. Ghép lại tập TRAIN & TEST cho mô hình
# ------------------------------------------------------------

train_df = pd.concat([ls_train_A, nonls_train_A], ignore_index=True)
test_df  = pd.concat([ls_test, nonls_test],      ignore_index=True)

train_df = train_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
test_df  = test_df.sample(frac=1.0,  random_state=random_state).reset_index(drop=True)

print("\n==> Tập cuối cùng:")
print("TRAIN total  :", train_df.shape)
print("TEST total   :", test_df.shape)

val_df = ls_train_B.copy()  # nếu cần dùng validation riêng cho LS

# ------------------------------------------------------------
# 6. SMOTENC + Normalize (tạo thêm 100 LS = 1)
# ------------------------------------------------------------

# Tách X, y dưới dạng DataFrame để lấy tên cột
X_train_df = train_df.drop(columns=[label_col])
y_train    = train_df[label_col].astype(int)

X_test_df  = test_df.drop(columns=[label_col])
y_test     = test_df[label_col].astype(int)

print("\nColumns in train_df:", list(X_train_df.columns))

# CÁC CỘT CATEGORICAL (sửa nếu tên khác trong Excel)
categorical_cols = [
    "Soil_type_",   # nhớ khớp tên column trong file
    "Geology_nu",
    "Forest_den",
    "Timper_age",
    "Diameter",
    "Forest_typ"
]

cat_idx = [X_train_df.columns.get_loc(c) for c in categorical_cols]

X_train_np = X_train_df.values
y_train_np = y_train.values

# số LS hiện có trong train
current_ls = np.sum(y_train_np == 1)
target_ls  = current_ls + 100   # tạo thêm 100 LS

smote = SMOTENC(
    categorical_features=cat_idx,
    sampling_strategy={1: target_ls},
    random_state=42,
    n_jobs=-1
)

X_train_res, y_train_res = smote.fit_resample(X_train_np, y_train_np)

print("Before SMOTE: shape =", X_train_np.shape,  "| class =", np.bincount(y_train_np))
print("After  SMOTE: shape =", X_train_res.shape, "| class =", np.bincount(y_train_res))

total_ls_after  = np.sum(y_train_res == 1)
new_ls_created  = total_ls_after - current_ls
print("New LS created:", new_ls_created)

# ------------------------------------------------------------
# 6.x Lấy đúng 100 điểm LS mới do SMOTENC tạo ra
# ------------------------------------------------------------

# Vị trí (index) của tất cả mẫu LS trong tập sau SMOTE
ls_indices_after = np.where(y_train_res == 1)[0]

# 100 mẫu LS mới chính là 100 index LS cuối cùng
synthetic_ls_indices = ls_indices_after[-new_ls_created:]   # new_ls_created = 100

# Tạo DataFrame chứa 100 điểm LS synthetic
synthetic_ls_df = pd.DataFrame(
    X_train_res[synthetic_ls_indices],
    columns=X_train_df.columns
)
synthetic_ls_df[label_col] = 1  # gắn nhãn lớp 1

print("\n========== 100 LANDSLIDE SYNTHETIC (SMOTENC) ==========")
print(synthetic_ls_df)

# Lưu ra Excel để kiểm tra / đưa vào GIS
synthetic_ls_df.to_excel("LS_2_aftercheck_SMOTE_100.xlsx", index=False)
print("\nSaved 100 new LS samples to: LS_2_aftercheck_SMOTE_100.xlsx")

# Sau đó vẫn dùng toàn bộ dữ liệu resampled cho train
X_train = X_train_res
y_train = y_train_res

# Test GIỮ NGUYÊN, không SMOTE
X_test = X_test_df.values
y_test = y_test.values



