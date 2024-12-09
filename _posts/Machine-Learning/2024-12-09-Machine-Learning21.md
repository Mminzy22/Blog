---
layout: post
title: "Machine Learning 21: 도미와 빙어 데이터를 활용한 지도 학습 실습"
date: 2024-12-09
categories: [Machine Learning] 
---


머신러닝은 데이터를 기반으로 패턴을 학습하고 예측하는 기술입니다. 오늘은 **지도 학습(Supervised Learning)**의 대표적인 알고리즘인 **K-최근접 이웃(K-Nearest Neighbors, KNN)**을 실습하면서 기본 개념과 과정을 살펴보겠습니다.


### 🎯 머신러닝 학습 유형 요약

1. **지도 학습 (Supervised Learning)**
   - 입력 데이터와 정답(타깃)을 제공하여 모델을 학습합니다.
   - 목표: 새로운 데이터에 대한 정답을 예측.
   - 대표 알고리즘: K-최근접 이웃(KNN), 선형 회귀, 의사결정 트리 등.

2. **비지도 학습 (Unsupervised Learning)**
   - 정답(타깃) 없이 데이터만 사용하여 패턴을 학습합니다.
   - 목표: 데이터의 특징 또는 구조를 이해.
   - 대표 알고리즘: K-평균 군집화, PCA(주성분 분석) 등.


### 🐟 1. 데이터 준비 및 시각화

#### 도미와 빙어 데이터
머신러닝 모델은 **특성(feature)**과 **타깃(target)** 데이터를 사용해 학습합니다. 이번 실습에서는 생선의 **길이**와 **무게**를 특성으로, 도미(1)와 빙어(0)를 타깃으로 사용합니다.

```python
fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
               31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
               35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8,
               10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
               500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
               700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7,
               7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]
```

#### 데이터 시각화
`Matplotlib`를 사용해 도미와 빙어 데이터를 산점도로 표현합니다.

```python
import matplotlib.pyplot as plt

plt.scatter(fish_length[:35], fish_weight[:35], label="Bream (도미)", color='blue')
plt.scatter(fish_length[35:], fish_weight[35:], label="Smelt (빙어)", color='red')
plt.xlabel('Length (cm)')
plt.ylabel('Weight (g)')
plt.legend()
plt.title('Bream and Smelt Data')
plt.show()
```

**결과:** 도미는 길이와 무게가 비례하는 관계를 보이지만, 빙어는 그렇지 않습니다.
![결과 산점도]({{ site.baseurl }}/assets/images/2024-12-09_산점도.png)


### 🤖 2. 데이터 준비와 모델 학습

#### 데이터 전처리
특성 데이터를 2차원 배열로 변환하고, 타깃 데이터를 설정합니다.

```python
import numpy as np

fish_data = [[l, w] for l, w in zip(fish_length, fish_weight)]
fish_target = [1]*35 + [0]*14  # 도미=1, 빙어=0
```

#### 데이터 섞기
랜덤하게 데이터를 섞어 샘플링 편향 문제를 방지합니다.

```python
np.random.seed(42)
index = np.arange(len(fish_data))
np.random.shuffle(index)

fish_data = np.array(fish_data)[index]
fish_target = np.array(fish_target)[index]
```

#### 데이터 분리
훈련 세트와 테스트 세트로 데이터를 나눕니다.

```python
train_data = fish_data[:35]
train_target = fish_target[:35]
test_data = fish_data[35:]
test_target = fish_target[35:]
```

#### 모델 학습
`KNeighborsClassifier`를 사용해 K-최근접 이웃 모델을 학습시킵니다.

```python
from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier(n_neighbors=5)  # 이웃 수: 5
kn.fit(train_data, train_target)  # 모델 학습
```


### 🔍 3. 모델 평가 및 예측

#### 모델 평가
훈련된 모델을 테스트 세트로 평가합니다.

```python
accuracy = kn.score(test_data, test_target)
print("테스트 세트 정확도:", accuracy)
```

#### 새로운 데이터 예측
길이 30cm, 무게 600g인 생선의 종류를 예측합니다.

```python
prediction = kn.predict([[30, 600]])
print("예측 결과:", "도미" if prediction[0] == 1 else "빙어")
```


### 🛠️ 4. 핵심 패키지와 함수

#### 1️⃣ NumPy
- `np.random.seed(n)`: 난수 생성 초깃값을 설정해 동일한 난수를 재현할 수 있습니다.
- `np.arange(start, stop, step)`: 지정된 간격으로 정수 또는 실수 배열을 생성합니다.
- `np.random.shuffle(arr)`: 배열을 무작위로 섞습니다. 다차원 배열일 경우 첫 번째 축(행)만 섞습니다.

#### 2️⃣ scikit-learn
- **KNeighborsClassifier()**: K-최근접 이웃 분류 모델을 생성하는 클래스입니다.
  - `fit(X, y)`: 모델 학습.
  - `predict(X)`: 새로운 데이터 예측.
  - `score(X, y)`: 모델 성능 평가.


### 🌟 정리

1. **지도 학습**은 입력과 정답 데이터를 사용해 모델을 학습한 후, 새로운 데이터를 예측합니다.
2. **훈련 세트**는 모델 학습에 사용되며, **테스트 세트**는 모델 평가에 사용됩니다. 
3. 샘플링 편향을 방지하기 위해 데이터를 무작위로 섞는 것이 중요합니다.
4. K-최근접 이웃 알고리즘은 간단하면서도 강력한 지도 학습 알고리즘입니다.
