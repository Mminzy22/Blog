---
layout: post
title: "Machine Learning 31: 비지도 학습과 군집화 알고리즘"
date: 2024-12-14
categories: [Machine Learning]
tags: [군집화, 비지도 학습, 머신러닝, 이미지 처리, 데이터 분석, 파이썬, 넘파이, 맷플롯립]
---


오늘은 머신러닝에서 **비지도 학습**의 핵심 주제 중 하나인 **군집화(Clustering)** 알고리즘을 배우며, 이를 실제 데이터에 적용해 보았습니다. 특히, 흑백 과일 이미지 데이터셋을 활용하여 군집화 과정을 단계별로 살펴보았습니다.


### 1. 군집화(Clustering)란 무엇인가?

**군집화**는 데이터 간의 유사성을 기준으로 그룹(군집)을 형성하는 비지도 학습 기법입니다.  
레이블(정답)이 없는 데이터에서 비슷한 특징을 가진 데이터끼리 묶는 데 사용됩니다.  
예를 들어, 다양한 과일 이미지에서 비슷한 모양이나 밝기를 기준으로 **사과**, **파인애플**, **바나나**와 같은 군집으로 나눌 수 있습니다.


### 2. 사용 데이터 소개

이번 실습에서는 100x100 픽셀 크기의 흑백 과일 이미지 데이터셋 `fruits_300.npy`를 사용했습니다.  
이 데이터셋은 **사과(100개)**, **파인애플(100개)**, **바나나(100개)**로 구성되어 있으며, 총 300개의 이미지가 포함되어 있습니다.


### 3. 데이터 불러오기 및 구조 확인

우선 데이터셋을 불러온 후 구조를 확인합니다.

```python
import numpy as np
import matplotlib.pyplot as plt

# 데이터 로드
fruits = np.load('fruits_300.npy')

# 데이터 구조 확인
print(fruits.shape)
```

출력 결과:

```
(300, 100, 100)
```

**해석**: 데이터는 총 300개의 이미지로 구성되어 있으며, 각 이미지의 크기는 100x100입니다.


### 4. 데이터 시각화

**시각화**는 데이터의 특성을 이해하는 데 필수적인 과정입니다. 

#### (1) 단일 이미지 확인

```python
plt.imshow(fruits[0], cmap='gray')
plt.show()

plt.imshow(fruits[0], cmap='gray_r')
plt.show()
```

- 첫 번째 이미지를 기본 흑백(`gray`)과 반전된 흑백(`gray_r`)으로 시각화했습니다.  
- 이를 통해 데이터셋에 저장된 이미지의 밝기와 패턴을 확인할 수 있습니다.

#### (2) 여러 이미지 비교

```python
fig, axs = plt.subplots(1, 2)
axs[0].imshow(fruits[100], cmap='gray_r')
axs[1].imshow(fruits[200], cmap='gray_r')
plt.show()
```

- 위 코드는 101번째 이미지(파인애플)와 201번째 이미지(바나나)를 시각적으로 비교하는 예제입니다.  
- 서로 다른 과일 이미지의 특징을 확인할 수 있습니다.


### 5. 데이터 전처리 - 과일별 평균 밝기 계산

#### (1) 과일별 데이터 분리
이미지를 사과, 파인애플, 바나나로 나누고, 각 이미지를 1차원 배열로 변환합니다.

```python
apple = fruits[0:100].reshape(-1, 100*100)
pineapple = fruits[100:200].reshape(-1, 100*100)
banana = fruits[200:300].reshape(-1, 100*100)
```

#### (2) 과일별 평균 밝기 확인

```python
print(apple.shape)
print(np.mean(apple, axis=1))
```

- 각 이미지의 밝기 평균값을 계산했습니다.  
- 이 값은 각 과일의 대표적인 특징을 나타내는 지표로 사용할 수 있습니다.


### 6. 히스토그램을 활용한 군집 분석

과일별 밝기 평균값의 분포를 히스토그램으로 시각화합니다.

```python
plt.hist(np.mean(apple, axis=1), alpha=0.8)
plt.hist(np.mean(pineapple, axis=1), alpha=0.8)
plt.hist(np.mean(banana, axis=1), alpha=0.8)
plt.legend(['apple', 'pineapple', 'banana'])
plt.show()
```

**결과 해석**:
- 사과, 파인애플, 바나나의 밝기 평균값 분포가 서로 다르며, 이를 기준으로 군집을 나눌 수 있습니다.


### 7. 군집별 평균 이미지 계산

각 과일 이미지를 평균내어 대표 이미지를 생성하고 이를 시각화했습니다.

```python
apple_mean = np.mean(apple, axis=0).reshape(100, 100)
pineapple_mean = np.mean(pineapple, axis=0).reshape(100, 100)
banana_mean = np.mean(banana, axis=0).reshape(100, 100)

fig, axs = plt.subplots(1, 3, figsize=(20, 5))
axs[0].imshow(apple_mean, cmap='gray_r')
axs[1].imshow(pineapple_mean, cmap='gray_r')
axs[2].imshow(banana_mean, cmap='gray_r')
plt.show()
```

**결과 해석**:
- **사과**: 둥글고 어두운 외곽선.
- **파인애플**: 균일한 밝기의 원형 이미지.
- **바나나**: 둥글면서 약간 찌그러진 특징.


### 8. 군집화 알고리즘 활용 - 유사 이미지 찾기

군집화 알고리즘을 적용하여 특정 과일과 가장 유사한 이미지를 찾을 수 있습니다.

#### (1) 사과와의 유사성 평가

```python
abs_diff = np.abs(fruits - apple_mean)
abs_mean = np.mean(abs_diff, axis=(1,2))
apple_index = np.argsort(abs_mean)[:100]

fig, axs = plt.subplots(10, 10, figsize=(10,10))
for i in range(10):
    for j in range(10):
        axs[i, j].imshow(fruits[apple_index[i*10 + j]], cmap='gray_r')
        axs[i, j].axis('off')
plt.show()
```

- 위 코드는 사과와 가장 유사한 100개의 이미지를 선택하여 10x10 그리드로 표시합니다.

#### (2) 바나나와의 유사성 평가

```python
abs_diff = np.abs(fruits - banana_mean)
abs_mean = np.mean(abs_diff, axis=(1,2))
banana_index = np.argsort(abs_mean)[:100]

fig, axs = plt.subplots(10, 10, figsize=(10,10))
for i in range(10):
    for j in range(10):
        axs[i, j].imshow(fruits[banana_index[i*10 + j]], cmap='gray_r')
        axs[i, j].axis('off')
plt.show()
```

- 동일한 방식으로 바나나와 유사한 이미지를 시각화합니다.


### 9. 결론

오늘 배운 군집화 알고리즘을 통해 다음과 같은 과정을 수행했습니다:
1. 데이터를 분류하지 않고 유사한 패턴을 기준으로 그룹화하는 방법.
2. 데이터 전처리를 통해 과일별 특징(밝기 평균값)을 분석.
3. 특정 과일과 유사한 이미지를 군집화하여 시각적으로 확인.

군집화는 비슷한 데이터를 자동으로 묶는 강력한 비지도 학습 기법입니다. 이번 실습에서는 간단한 과일 이미지 데이터셋을 사용했지만, 이 기법은 고객 세분화, 문서 분류, 이미지 검색 등 다양한 분야에서 활용될 수 있습니다.
