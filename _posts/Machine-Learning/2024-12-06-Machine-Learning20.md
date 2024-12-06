---
layout: post
title: "Machine Learning 20: 도미와 빙어 데이터를 활용한 머신러닝 학습"
date: 2024-12-06
categories: [Machine Learning] 
---


머신러닝을 처음 시작할 때는 간단한 데이터셋으로 이해하기 쉬운 알고리즘부터 학습하는 것이 좋습니다. 오늘은 **도미와 빙어 데이터**를 활용해 머신러닝 알고리즘 중 하나인 **k-최근접 이웃(KNN)**을 학습하고 이를 통해 데이터를 분류하는 방법을 살펴보겠습니다. 


### 🐟 1. 도미 데이터 준비하기

도미 데이터를 **길이(length)**와 **무게(weight)**라는 두 가지 특성으로 표현합니다. 아래는 도미 데이터의 예시입니다.

```python
bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]
```


### 📊 2. 도미 데이터 시각화

도미 데이터를 **산점도(scatter plot)**로 그려 두 특성 간의 관계를 시각적으로 확인해봅니다.

```python
import matplotlib.pyplot as plt

plt.scatter(bream_length, bream_weight, label="Bream", color='blue')
plt.xlabel('Length (cm)')
plt.ylabel('Weight (g)')
plt.title('Bream Data')
plt.legend()
plt.show()
```

💡 **결과:** 산점도는 **선형적 관계**를 보여줍니다. 즉, 도미의 길이가 증가할수록 무게도 비례하여 증가합니다.


### ❄️ 3. 빙어 데이터 준비 및 시각화

빙어 데이터도 **길이**와 **무게** 특성을 가지고 있습니다. 이를 도미 데이터와 함께 비교하여 시각화해보겠습니다.

```python
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

plt.scatter(bream_length, bream_weight, label="Bream", color='blue')
plt.scatter(smelt_length, smelt_weight, label="Smelt", color='red')
plt.xlabel('Length (cm)')
plt.ylabel('Weight (g)')
plt.title('Bream and Smelt Data')
plt.legend()
plt.show()
```

💡 **결과:** 빙어는 도미와 달리 길이가 증가해도 무게가 크게 증가하지 않음을 확인할 수 있습니다.


### 🤖 4. 머신러닝 알고리즘 적용

#### 4.1 데이터 준비
도미와 빙어 데이터를 **특성 데이터**와 **정답 데이터**로 분리합니다.

```python
fish_length = bream_length + smelt_length
fish_weight = bream_weight + smelt_weight
fish_data = [[l, w] for l, w in zip(fish_length, fish_weight)]
fish_target = [1] * 35 + [0] * 14  # 도미=1, 빙어=0
```

#### 4.2 모델 학습
사이킷런의 `KNeighborsClassifier`를 이용해 **k-최근접 이웃 알고리즘**을 학습시킵니다.

```python
from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier()
kn.fit(fish_data, fish_target)
```

#### 4.3 모델 평가
모델의 정확도를 확인합니다.

```python
kn.score(fish_data, fish_target)
# 출력: 1.0
```

💡 **결과:** 정확도 100%로 도미와 빙어를 완벽히 분류합니다.


### 🧠 5. 새로운 데이터 예측

길이 30cm, 무게 600g인 생선이 도미인지 빙어인지 예측해봅니다.

```python
kn.predict([[30, 600]])
# 출력: array([1])
```

💡 **결과:** 해당 생선은 **도미**로 분류되었습니다.


### 🌟 6. 추가 도전 과제

#### ✅ 이웃 수 조정해보기
`n_neighbors` 매개변수를 조정하여 모델의 성능 변화를 확인하세요.

```python
kn = KNeighborsClassifier(n_neighbors=10)
kn.fit(fish_data, fish_target)
print(kn.score(fish_data, fish_target))
```

#### ✅ 데이터를 학습/테스트로 나누기
전체 데이터를 **학습용**과 **테스트용**으로 나눠 모델의 일반화 성능을 평가해보세요.

```python
from sklearn.model_selection import train_test_split

train_data, test_data, train_target, test_target = train_test_split(
    fish_data, fish_target, test_size=0.3, random_state=42
)
kn.fit(train_data, train_target)
print("Test accuracy:", kn.score(test_data, test_target))
```

#### ✅ 다른 알고리즘 사용해보기
로지스틱 회귀, 서포트 벡터 머신(SVM) 등 다른 분류 알고리즘을 사용해 성능을 비교해보세요.


### 🌟 7. 정리

1. **특성(feature)**: 데이터를 표현하는 하나의 성질입니다. 예를 들어, 생선 데이터를 길이와 무게로 나타낼 수 있습니다.
2. **훈련(training)**: 머신러닝 알고리즘이 데이터에서 규칙을 찾는 과정을 의미합니다. 사이킷런에서는 `fit()` 메서드가 이 역할을 합니다.
3. **k-최근접 이웃 알고리즘(KNN)**: 가장 간단한 머신러닝 알고리즘 중 하나로, 새로운 데이터를 분류할 때 주변 데이터를 참고합니다. 규칙을 학습하기보다는 데이터를 메모리에 저장하여 사용합니다.
4. **모델(model)**: 머신러닝 프로그램에서 알고리즘이 구현된 객체를 의미하며, 학습된 알고리즘 자체를 가리키기도 합니다.
5. **정확도(accuracy)**: 예측이 얼마나 정확했는지 백분율로 나타낸 값입니다.  
   정확도 = (정확히 맞힌 개수) / (전체 데이터 개수)


### 🛠️ 8. 핵심 패키지와 함수

#### 1️⃣ Matplotlib
- `scatter(x, y)`: 산점도를 그리는 함수입니다.  
  - `x`: x축 데이터  
  - `y`: y축 데이터  
  - `color`, `marker` 등 다양한 스타일 지정 가능

#### 2️⃣ scikit-learn
- **KNeighborsClassifier()**: k-최근접 이웃 분류 모델을 만드는 클래스입니다.  
  - `n_neighbors`: 참고할 이웃의 개수 (기본값: 5)  
  - `fit(X, y)`: 모델을 훈련하는 메서드입니다.  
  - `predict(X)`: 새로운 데이터를 예측합니다.  
  - `score(X, y)`: 모델의 성능(정확도)을 측정합니다.
