import pandas as pd

# 引入 Normalizer 進行向量歸一化 (Vector Normalization)
from sklearn.preprocessing import Normalizer
import numpy as np

# 假設 'final' DataFrame 已經載入
data = {
    "age": [
        30,
        33,
        35,
        30,
        59,
    ],
    "balance": [
        1787,
        4789,
        1350,
        1476,
        0,
    ],
    "loan": [0, 1, 0, 1, 0],
    "y": [
        0,
        0,
        0,
        0,
        0,
    ],
}
final = pd.DataFrame(data)

# --- Normalizer (L2 向量歸一化) 步驟 ---

# 1. 初始化 Normalizer (預設 norm='l2'，即歐幾里得長度為 1)
normalizer = Normalizer(norm="l2")

# 2. 選擇要歸一化的特徵
# 注意：這裡是對每一列的 [age, balance] 兩個值組成的向量進行歸一化。
features_to_normalize = final[["age", "balance"]]

# 3. 進行歸一化並將結果轉換回 DataFrame
normalized_array = normalizer.fit_transform(features_to_normalize)
final_normalized = pd.DataFrame(
    normalized_array, columns=["age_normalized", "balance_normalized"]
)

print("--- Normalizer (L2) 歸一化後資料的前五行 ---")
print(final_normalized.head())
# 額外檢查：可以看到每一行 (age_normalized^2 + balance_normalized^2) 都約等於 1
# print((final_normalized.iloc[0]['age_normalized']**2 + final_normalized.iloc[0]['balance_normalized']**2)**0.5)

# --- 變異數 (Variance) 比較步驟 ---

# 4. 計算歸一化後特徵的變異數
variance_age_normalized = final_normalized["age_normalized"].var()
variance_balance_normalized = final_normalized["balance_normalized"].var()

print("\n--- 歸一化後特徵的變異數比較 ---")
print(f"age_normalized 的變異數 (Variance): {variance_age_normalized:.10f}")
print(f"balance_normalized 的變異數 (Variance): {variance_balance_normalized:.10f}")

# 5. 比較哪個變異數較低
if variance_age_normalized < variance_balance_normalized:
    result = "age"
elif variance_balance_normalized < variance_age_normalized:
    result = "balance"
else:
    result = "兩者的變異數相等 (或非常接近)"

print(f"\n結論：在 Normalizer (L2 向量歸一化) 之後，{result} feature 的變異數比較低。")

# 注意：在這個例子中，由於 'balance' 的原始數值通常遠大於 'age'，
# Normalizer 處理後，'balance_normalized' 的數值會更集中在 1 附近，
# 而 'age_normalized' 的數值會更集中在 0 附近，
# 因此 'balance_normalized' 的變異數預期會低於 'age_normalized'。
# (變異數較低，代表數值比較集中)
