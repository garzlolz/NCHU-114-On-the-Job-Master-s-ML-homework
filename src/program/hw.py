import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    LassoCV,
    RANSACRegressor,
)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings("ignore")

# ========== 步驟 1: 載入與預處理所有特徵 ==========
print("=" * 80)
print("步驟 1: 資料載入與預處理")
print("=" * 80)

data = pd.read_csv("src/data/bank/bank.csv", sep=";")
print(f"原始資料形狀: {data.shape}")
print(f"欄位名稱: {list(data.columns)}")
print("\n前 5 筆資料:")
print(data.head())

# 檢查遺失值
print(f"\n遺失值統計:")
print(data.isnull().sum())

# 分離目標變數 - 修正欄位名稱為 'y'
y = data["y"].map({"yes": 1, "no": 0})
X = data.drop("y", axis=1)

# 處理類別變數
print("\n處理類別變數...")
categorical_columns = X.select_dtypes(include=["object"]).columns
numerical_columns = X.select_dtypes(include=["int64", "float64"]).columns

print(f"類別變數: {list(categorical_columns)}")
print(f"數值變數: {list(numerical_columns)}")

# One-Hot Encoding 類別變數
X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
print(f"\nOne-Hot Encoding 後的特徵數量: {X_encoded.shape[1]}")

# 標準化數值特徵
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_encoded), columns=X_encoded.columns)

print(f"標準化後的資料形狀: {X_scaled.shape}")

# ========== 步驟 2: 應用技巧進行特徵選擇/降維 ==========
print("\n" + "=" * 80)
print("步驟 2: 特徵選擇與降維")
print("=" * 80)

# 2.1 低變異度過濾器
print("\n2.1 低變異度過濾器")
variance_threshold = VarianceThreshold(threshold=0.01)
X_variance = variance_threshold.fit_transform(X_scaled)
selected_features_variance = X_scaled.columns[variance_threshold.get_support()].tolist()
print(f"低變異度過濾後保留特徵數: {len(selected_features_variance)}")

# 2.2 高相關性過濾器
print("\n2.2 高相關性過濾器")
X_variance_df = pd.DataFrame(X_variance, columns=selected_features_variance)
correlation_matrix = X_variance_df.corr().abs()
upper_triangle = correlation_matrix.where(
    np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
)
high_corr_features = [
    column for column in upper_triangle.columns if any(upper_triangle[column] > 0.8)
]
X_low_corr = X_variance_df.drop(columns=high_corr_features)
print(f"移除高相關特徵後保留: {X_low_corr.shape[1]} 個特徵")
print(f"移除的高相關特徵數量: {len(high_corr_features)}")

# 2.3 SelectKBest 統計檢定
print("\n2.3 SelectKBest 統計檢定")
k_best = min(20, X_low_corr.shape[1])  # 選擇最多 20 個特徵
selector = SelectKBest(score_func=f_classif, k=k_best)
X_selected = selector.fit_transform(X_low_corr, y)
selected_features_kbest = X_low_corr.columns[selector.get_support()].tolist()
print(f"SelectKBest 保留前 {k_best} 個特徵")

# 2.4 Lasso 特徵選擇
print("\n2.4 Lasso L1 正規化特徵選擇")
lasso_cv = LassoCV(cv=5, random_state=42, max_iter=10000)
lasso_cv.fit(X_low_corr, y)
print(f"最佳 alpha: {lasso_cv.alpha_}")

lasso_selector = SelectFromModel(lasso_cv, prefit=True, threshold=1e-5)
X_lasso = lasso_selector.transform(X_low_corr)
selected_features_lasso = X_low_corr.columns[lasso_selector.get_support()].tolist()
print(f"Lasso 選擇的特徵數: {len(selected_features_lasso)}")

# 使用 Lasso 選擇的特徵作為最終特徵集
X_final = pd.DataFrame(X_lasso, columns=selected_features_lasso)
print(f"\n最終特徵集形狀: {X_final.shape}")
print(f"選擇的特徵: {selected_features_lasso}")

# ========== 步驟 3: 資料輸出與分割 ==========
print("\n" + "=" * 80)
print("步驟 3: 資料輸出與分割")
print("=" * 80)

# 輸出為 CSV
output_data = X_final.copy()
output_data["deposit"] = y.values
output_path = "src/data/bank/bank_tech_features.csv"
output_data.to_csv(output_path, index=False)
print(f"已輸出處理後資料至: {output_path}")

# 分割訓練集與測試集
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42, stratify=y
)
print(f"訓練集大小: {X_train.shape}")
print(f"測試集大小: {X_test.shape}")

# ========== 第二部分: 線性迴歸預測 ==========
print("\n" + "=" * 80)
print("第二部分: 線性迴歸預測")
print("=" * 80)

# 1. 純線性機制預測
print("\n1. 純線性迴歸")
print("-" * 40)

# 1.1 LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print(f"LinearRegression - R² Score: {r2_score(y_test, y_pred_lr):.4f}")
print(f"LinearRegression - MSE: {mean_squared_error(y_test, y_pred_lr):.4f}")
print(f"LinearRegression - MAE: {mean_absolute_error(y_test, y_pred_lr):.4f}")

# 1.2 Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)
print(f"\nRidge Regression - R² Score: {r2_score(y_test, y_pred_ridge):.4f}")
print(f"Ridge Regression - MSE: {mean_squared_error(y_test, y_pred_ridge):.4f}")
print(f"Ridge Regression - MAE: {mean_absolute_error(y_test, y_pred_ridge):.4f}")

# 1.3 Lasso Regression
lasso = Lasso(alpha=0.01, max_iter=10000)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)
print(f"\nLasso Regression - R² Score: {r2_score(y_test, y_pred_lasso):.4f}")
print(f"Lasso Regression - MSE: {mean_squared_error(y_test, y_pred_lasso):.4f}")
print(f"Lasso Regression - MAE: {mean_absolute_error(y_test, y_pred_lasso):.4f}")

# 1.4 RANSAC Regressor (處理離群值)
ransac = RANSACRegressor(LinearRegression(), random_state=42)
ransac.fit(X_train, y_train)
y_pred_ransac = ransac.predict(X_test)
print(f"\nRANSAC Regressor - R² Score: {r2_score(y_test, y_pred_ransac):.4f}")
print(f"RANSAC Regressor - MSE: {mean_squared_error(y_test, y_pred_ransac):.4f}")
print(f"RANSAC Regressor - MAE: {mean_absolute_error(y_test, y_pred_ransac):.4f}")

# 2. 非線性機制預測(多項式迴歸)
print("\n" + "=" * 80)
print("2. 多項式迴歸 (非線性)")
print("-" * 40)

# 使用 degree=2 的多項式特徵
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

print(f"多項式特徵擴展後的訓練集形狀: {X_train_poly.shape}")
print(f"多項式特徵擴展後的測試集形狀: {X_test_poly.shape}")

# 2.1 多項式 + LinearRegression
lr_poly = LinearRegression()
lr_poly.fit(X_train_poly, y_train)
y_pred_poly = lr_poly.predict(X_test_poly)
print(f"\n多項式 LinearRegression - R² Score: {r2_score(y_test, y_pred_poly):.4f}")
print(f"多項式 LinearRegression - MSE: {mean_squared_error(y_test, y_pred_poly):.4f}")
print(f"多項式 LinearRegression - MAE: {mean_absolute_error(y_test, y_pred_poly):.4f}")

# 2.2 多項式 + Ridge (避免過擬合)
ridge_poly = Ridge(alpha=10.0)
ridge_poly.fit(X_train_poly, y_train)
y_pred_ridge_poly = ridge_poly.predict(X_test_poly)
print(
    f"\n多項式 Ridge Regression - R² Score: {r2_score(y_test, y_pred_ridge_poly):.4f}"
)
print(
    f"多項式 Ridge Regression - MSE: {mean_squared_error(y_test, y_pred_ridge_poly):.4f}"
)
print(
    f"多項式 Ridge Regression - MAE: {mean_absolute_error(y_test, y_pred_ridge_poly):.4f}"
)

# ========== 總結 ==========
print("\n" + "=" * 80)
print("作業完成總結")
print("=" * 80)
print(f"✓ 原始特徵數: {X_scaled.shape[1]}")
print(f"✓ 最終選擇特徵數: {X_final.shape[1]}")
print(f"✓ 輸出檔案: {output_path}")
print(f"✓ 訓練/測試分割比例: 80/20")
print(f"✓ 已完成線性迴歸 (4種) 和多項式迴歸 (2種) 模型訓練與評估")
print("=" * 80)
