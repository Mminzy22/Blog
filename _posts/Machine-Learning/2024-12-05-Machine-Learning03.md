---
layout: post
title: "Machine Learning: 데이터 전처리 방법"
date: 2024-12-05
categories: [Machine Learning] 
---

데이터 전처리는 머신러닝과 데이터 분석에서 필수적인 단계로, 원시 데이터를 정제하고 분석 또는 모델 학습에 적합한 형태로 준비하는 과정입니다. 이번 글에서는 데이터 전처리의 다양한 기법과 코드 예제를 통해 데이터를 효율적으로 처리하는 방법을 알아보겠습니다.


#### **1. 결측값 처리**

**결측값이란?**  
데이터셋에서 값이 누락된 부분을 의미하며, 처리하지 않으면 분석과 모델 성능에 악영향을 미칩니다.

- **제거 방법**  
  ```python
  # 결측값이 포함된 행 제거
  df_dropped_rows = df.dropna()

  # 결측값이 포함된 열 제거
  df_dropped_cols = df.dropna(axis=1)
  ```
  
- **대체 방법**  
  ```python
  # 결측값을 0으로 대체
  df_filled = df.fillna(0)

  # 각 열의 평균값으로 대체
  df_filled_mean = df.fillna(df.mean())

  # 각 열의 중간값으로 대체
  df_filled_median = df.fillna(df.median())
  ```

- **예측을 통한 대체**  
  머신러닝 모델을 사용하여 결측값을 예측하고 채울 수 있습니다.  
  ```python
  from sklearn.linear_model import LinearRegression

  # 회귀 모델 학습 및 결측값 예측
  model = LinearRegression()
  model.fit(df_without_na[['feature1', 'feature2']], df_without_na['column_with_na'])
  predicted_values = model.predict(df_with_na[['feature1', 'feature2']])
  df.loc[df['column_with_na'].isnull(), 'column_with_na'] = predicted_values
  ```


#### **2. 이상치 처리**

**이상치란?**  
비정상적으로 크거나 작은 값으로, 모델 성능에 부정적인 영향을 미칠 수 있습니다.

- **이상치 탐지 (IQR 방법)**  
  ```python
  Q1 = df['column_name'].quantile(0.25)
  Q3 = df['column_name'].quantile(0.75)
  IQR = Q3 - Q1
  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR

  # 이상치 확인
  outliers = df[(df['column_name'] < lower_bound) | (df['column_name'] > upper_bound)]
  ```

- **이상치 처리**  
  ```python
  # 이상치 제거
  df_no_outliers = df[(df['column_name'] >= lower_bound) & (df['column_name'] <= upper_bound)]

  # 이상치를 평균값으로 대체
  mean_value = df['column_name'].mean()
  df['column_name'] = df['column_name'].apply(lambda x: mean_value if x < lower_bound or x > upper_bound else x)
  ```


#### **3. 중복값 제거**

**중복값이란?**  
같은 데이터가 반복된 경우를 의미하며, 이를 제거하여 데이터의 신뢰성을 높일 수 있습니다.  
```python
# 중복된 행 확인
print(df.duplicated().sum())

# 중복된 행 제거
df_no_duplicates = df.drop_duplicates()
```


#### **4. 데이터 타입 변환**

**필요성**  
잘못된 데이터 타입은 분석 결과와 모델 학습에 오류를 발생시킬 수 있습니다.  

- **데이터 타입 변환 방법**  
  ```python
  # 특정 열을 정수형으로 변환
  df['column_name'] = df['column_name'].astype(int)

  # 특정 열을 문자열로 변환
  df['column_name'] = df['column_name'].astype(str)

  # 특정 열을 부동 소수점으로 변환
  df['column_name'] = df['column_name'].astype(float)
  ```


#### **5. 인코딩**

**인코딩이란?**  
범주형 데이터를 수치형 데이터로 변환하는 과정입니다. 머신러닝 모델은 수치형 데이터를 입력으로 받기 때문에 필수적입니다.

- **범주형 데이터의 원-핫 인코딩**  
  ```python
  # 범주형 데이터를 더미 변수로 변환
  df_encoded = pd.get_dummies(df, columns=['category_column'])
  print(df_encoded.head())
  ```


#### **6. 샘플링**

**샘플링이란?**  
데이터의 크기를 조절하거나 대표성을 유지하면서 샘플을 추출하는 과정입니다.  

- **샘플 추출**  
  ```python
  # 데이터셋에서 50% 샘플 추출
  df_sampled = df.sample(frac=0.5)

  # 데이터셋에서 100개의 샘플 추출
  df_sampled_n = df.sample(n=100)
  ```


#### **7. 특징 선택 및 추출**

**특징 선택과 추출**  
데이터의 차원을 줄이거나 새로운 유용한 특징을 생성하여 모델의 성능을 높이는 기법입니다.

- **특징 선택**  
  ```python
  from sklearn.feature_selection import SelectKBest, f_classif

  # 상위 5개의 중요한 특징 선택
  selector = SelectKBest(score_func=f_classif, k=5)
  X_new = selector.fit_transform(X, y)
  selected_features = selector.get_support(indices=True)
  print(selected_features)
  ```

- **특징 생성**  
  ```python
  # 새로운 특징 생성 (두 열의 곱)
  df['new_feature'] = df['feature1'] * df['feature2']
  ```
