---
applyTo: "**"
---

## 題目

<!-- 請以 Bank 為範例進行資料前處理（分成兩部分：一是由以下feature selection方法進行選擇 features「要說明理由」，一是由以下講義的技巧來進行）。整理後的資料，請以 CSV 的格式輸出（自己找答案），上傳並留待下一個作業。

以上述前處理之後的資料集，以 linear regression （至少包含線性以及非線性的兩種機制）進行預測。

## 講義中使用到的 alogrithm

目前已用到的機器學習演算法，可以主要分為線性（Linear）和非線性（Non-Linear）兩大類。

以下是根據您提供的資料，已提及或使用的演算法說明：

---

### 一、 線性演算法 (Linear Algorithms)

線性模型是機器學習中最簡單的參數化方法，即使是內在非線性的問題，也常能利用這些模型輕鬆解決。

#### 1. 線性回歸 (Linear Regression)

- **定義與目標：** 線性回歸是一種迴歸（Regression）方法，其目標是預測連續的目標變數。它利用「線」來推測資料的關係。
- **訓練過程：** 訓練的目標是找到一條 Loss 最小的線。在 `sklearn` 中，`LinearRegression` 預設使用的損失函數是**普通最小平方法 (Ordinary Least Square, OLS)**。
- **結果：** 經由訓練，模型會學到最佳參數 $\alpha_i$ (即截距 `intercept_` 和係數 `coef_`)。
- **應用：** 在一開始的銀行行銷案例中，曾嘗試選擇 `LinearRegression` 作為分類器。

#### 2. 具有正規化（Regularization）的線性回歸

為了避免模型在僅考慮誤差時發生**過度擬合 (overfitting)** 的現象，會引入正規化項（regularization term），藉此限制模型的複雜度。正規化項通常是係數向量 $\alpha_i$ 的範數（norm），常見的有 L1 範數或 L2 範數。

- **Ridge 回歸 (L2 Regularization):**

  - Ridge 回歸對 OLS 損失函數施加一個額外的 L2 範數（L2 norm）懲罰。
  - 這個懲罰項限制了權重 $w$ 的平方和，從而避免 $w$ 無限增長，這在資料存在多重共線性（multicollinearity）或病態條件（ill-conditioning）時特別有用。
  - **特點：** Ridge 會強制係數變小，但**不會**將不相關的特徵係數設為零。

- **Lasso 回歸 (L1 Regularization):**

  - Lasso 回歸對權重 $w$ 的 L1 範數施加懲罰。
  - **特點：** Lasso 不僅懲罰高係數值，還能將不相關特徵的係數**設為零**。因此，訓練後的 Lasso 模型通常是一個**稀疏模型 (sparse model)**，這也使其具有特徵選擇（feature selection）的能力。

- **ElasticNet:**
  - ElasticNet 結合了 Lasso (L1) 和 Ridge (L2) 兩種懲罰項。
  - **特點：** 由於 L1 和 L2 範數的平衡作用，所得的模型既能像 Lasso 一樣稀疏化，又能避免選擇性排除具有強相關性的特徵。

#### 3. 隨機抽樣一致性 (RANSAC)

- **目標：** RANSAC (Random Sample Consensus) 演算法本身是一種學習技術，用於估計模型參數，特別是當資料中存在**離群值 (outliers)** 時，用來估計模型（如線性回歸）的參數。
- **原理：** 它可以與線性回歸結合使用 (`RANSACRegressor`)，通過隨機抽樣資料子集（稱為假設內點，hypothetical inliers）來擬合模型，並將擬合良好的點歸類為共識集（consensus set），從而避免離群值對模型係數產生偏差（biased）影響。

---

### 二、 非線性演算法與非線性處理技術 (Non-Linear Algorithms and Techniques)

#### 1. 決策樹分類器 (Decision Tree Classifier)

- **特性：** 決策樹是一種非線性的分類器。
- **應用：** 在銀行行銷案例中，由於線性回歸效果不佳，因此嘗試選擇 `DecisionTreeClassifier`。
- **參數調整：** 決策樹的效能會受到許多超參數（hyperparameters）的影響，例如可以調整樹的深度（`max_depth`）。

#### 2. 多項式回歸 (Polynomial Regression)

- **技術本質：** 多項式回歸是一種**技巧**，它允許我們使用**線性模型**來處理具有強烈非線性特性的資料集。
- **原理：** 其核心思想是透過**特徵轉換** (Feature Transformation)，將原始特徵（如 $a, b$）擴展為高維度的多項式組合（例如 $1, a, b, a^2, b^2, ab$），從而將非線性問題轉換為在新的高維度空間中的線性問題。
- **實作：** 使用 `PolynomialFeatures` 類別來執行此類轉換。雖然特徵數量會增加，但如果增加的特徵數量超過可接受的閾值，則建議嘗試降維或者直接轉向非線性模型 (例如帶核函數的 SVM)。

## 講義中的資料前處理

根據您提供的資料，在處理銀行電話行銷（bank marketing）的案例中，對資料進行了以下主要的前處理步驟：

### 銀行資料集的前處理方法

該案例使用的是 UCI Machine Learning Deposit 中的 Bank Marketing Dataset。前處理（在資料準備階段，Data Preparation）主要涵蓋了特徵選擇和資料格式轉換的需求：

#### 1. 資料載入與檢查

首先，使用 `pandas` 函式庫讀取 `bank.csv` 檔案。

- **讀取設定：** 讀取時指定了分界符號 `delimiter=";"`，並設定 `header='infer'` 以自動讀取欄位名稱。

#### 2. 特徵選擇（Features Selection）

由於成本考量和機器學習目標，銀行內部人員討論後，決定僅採用與客戶是否進行定期存款（deposit, 即目標變數 $y$）最相關的幾個特徵。

- **選定特徵：** 最終保留的有用資料只有：**年齡（age）、存款餘額（balance）、貸款狀況（loan），以及目標變數（deposit, 即 $y$）**。
- **移除不相關特徵：** 透過 `data.drop` 方法，明確移除了資料集中所有被認為無用的特徵，包括 `job`、`marital`、`education`、`default`、`housing`、`contact`、`day`、`month`、`duration`、`campaign`、`pdays`、`previous` 和 `poutcome`。

#### 3. 處理非數值（Categorical）資料的需求

機器學習模型通常只處理數字，但在原始資料中，`loan` 和目標變數 `y` (deposit) 包含了非數值的「yes」和「no」資料。

- **轉換需求：** 因此，資料準備階段的一個重要步驟是必須先將 `loan` 和 `y` 的資料進行處理，轉換為機器可識別的數字格式（例如二元編碼）。

#### 4. 資料分割（Training/Testing Sets）

在選定並清理特徵後（生成 `final` 資料集），下一步是將資料集分割成訓練集（training set）和測試集（testing set），以供後續的模型訓練和評估。

- **分割方法：** 使用 `sklearn` 套件中的 `train_test_split` 函式來執行分割。
- **分割比例：** 在範例中，資料被分為 $X$（特徵）和 $y$（目標），並使用 `test_size = 0.2`（即 80% 訓練，20% 測試）進行分割。建議的分割比例通常是 8/2 或 7/3。

## 講義中的特徵選擇與降維方法總結

目前講義中提到了多種特徵選擇或降維的方法和技術：

### 一、 依據領域知識進行的特徵選擇（Domain Knowledge-based Feature Selection）

在實際的應用案例中，特徵選擇首先是基於對問題的理解和領域知識。

1.  **銀行電話行銷案例：**
    - 在銀行電話行銷的快速啟動範例中，銀行內部人員經過討論與細部分析後，**一致認同**可以採用 **貸款 (loan)**、**年齡 (age)**，以及 **存款餘額 (balance)** 來決定客戶是否會進行定期存款 (`deposit`)。
    - 因此，明確地移除了資料集中所有被認為無用的特徵，包括 `job`、`marital`、`education`、`default`、`housing`、`contact`、`day`、`month`、`duration`、`campaign`、`pdays`、`previous` 和 `poutcome`。

### 二、 基於過濾器（Filter-based）的特徵選擇方法

過濾器方法通常在模型訓練之前應用，透過資料的統計特性來篩選特徵。

1.  **低變異度過濾器 (Low Variance Filter)：**

    - 如果一個變數在所有觀察值上都具有相同的值（即變異數為零），那麼它對模型幾乎沒有幫助。因此，需要計算每個變數的變異數，並丟棄那些變異度相較於資料集中其他變數來說較低的變數。
    - 可以使用 `sklearn.feature_selection` 中的 `VarianceThreshold` 類別來設定變異數閾值進行特徵選擇。

2.  **高相關性過濾器 (High Correlation Filter)：**

    - 如果兩個變數之間存在高相關性（例如，相關係數大於 0.5 到 0.6），它們很可能傳遞相似的資訊，導致特徵冗餘。
    - 高相關性會嚴重降低某些模型（如線性或邏輯斯迴歸）的性能。因此，應考慮刪除其中一個變數。

3.  **互資訊 (Mutual Information)：**
    - 互資訊衡量兩個變數共享的資訊量。在特徵選擇中，可以衡量**特徵**與**類別標籤**之間的互資訊。如果互資訊很高，則該特徵是類別的強指標。
    - 如果特徵過多，也可以使用特徵之間的互資訊來移除冗餘特徵。

### 三、 基於嵌入器/包裝器（Embedded/Wrapper）的特徵選擇方法

這些方法依賴於機器學習模型的輸出來決定特徵的重要性。

1.  **特徵重要性過濾器 (Feature Importance Filter)：**

    - 這類方法是根據特徵的重要性來選擇特徵。
    - 常見的計算重要性的方法包括：**隨機森林 (Random Forest)**、**後向特徵消除 (Backward Feature Elimination)**（觀察移除單個特徵對性能的影響），以及**前向特徵消除 (Forward Feature Elimination)**（逐步加入最重要的特徵）。
    - `sklearn` 提供了 `SelectKBest`（選擇得分最高的 $K$ 個特徵）和 `SelectPercentile`（選擇屬於特定百分位數的特徵子集）。這些函式可以使用如 `f_regression`（基於統計檢定計算變數與目標的相關性）作為評分基準。

2.  **Lasso 回歸 (Lasso Regression, L1 Regularization)：**

    - Lasso 回歸在損失函數中施加了 L1 範數懲罰，其特點是不僅懲罰高係數值，還會將**不相關特徵的係數設為零**。
    - 因此，訓練後的 Lasso 模型通常是一個**稀疏模型 (sparse model)**，使其具有內建的特徵選擇能力。

3.  **使用 `SelectFromModel`：**
    - 可以利用 `SelectFromModel` 類別，讓 `scikit-learn` 根據模型訓練後估計的特徵重要性來選取最佳特徵（例如，用於多項式迴歸擴展後的特徵選擇）。

### 四、 降維技術（Dimensionality Reduction Techniques）

降維技術的目標是找到一組新的、維度較低的變數，這些新變數是原始變數的組合，但基本保留了原始變數所包含的資訊。

1.  **主成分分析 (Principal Component Analysis, PCA)：**

    - PCA 是一種技術，用於從現有的變數集中提取一組新的變數，稱為**主成分 (Principal Components)**。
    - 它旨在找到高維資料中最大變異度的方向，並將資料投影到較小的子空間，同時保留大部分資訊。
    - PCA 的核心步驟包括：標準化資料、計算協方差矩陣、提取特徵值和特徵向量，並選擇對應於最大特徵值的 $k$ 個特徵向量來構成投影矩陣。

2.  **其他降維方法：**
    - 資料中也提及可以參考 `sklearn.decomposition` 內的**因子分析 (FactorAnalysis)**、**非負矩陣分解 (NMF)** 和 **字典學習 (DictionaryLearning)**。 -->
