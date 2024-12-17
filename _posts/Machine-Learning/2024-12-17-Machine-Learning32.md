---
layout: post
title: "Machine Learning 32: K-Means Clustering: 이미지 데이터 분석 사례"
date: 2024-12-17
categories: [Machine Learning]
tags: [K-Means, 클러스터링, 머신러닝, 이미지 분석, 데이터 시각화, 파이썬, scikit-learn, 데이터 과학, 알고리즘, 인공지능]
---


이번 포스팅에서는 K-Means 클러스터링을 사용하여 이미지 데이터를 분석하는 과정을 소개합니다. 실습 데이터는 과일 사진 데이터이며, 이를 통해 클러스터링의 작동 원리를 단계별로 이해할 수 있습니다.


## 데이터 준비

먼저, 데이터를 다운로드하고 로드합니다. 데이터는 `fruits_300.npy` 파일로, 각 이미지는 100x100 크기의 흑백 사진으로 구성되어 있습니다.

```python
!wget https://bit.ly/fruits_300_data -O fruits_300.npy

import numpy as np

fruits = np.load('fruits_300.npy')
fruits_2d = fruits.reshape(-1, 100*100)
```

- `fruits`는 원본 이미지 데이터를 저장합니다.
- `fruits_2d`는 2D 배열로 변환된 데이터로, 각 이미지는 10,000(100x100) 차원의 벡터로 표현됩니다.


## K-Means 클러스터링 수행

K-Means 알고리즘을 사용하여 데이터를 3개의 그룹으로 나눕니다.

```python
from sklearn.cluster import KMeans

km = KMeans(n_clusters=3, random_state=42)
km.fit(fruits_2d)
```

### 클러스터 레이블 확인

클러스터링 결과 각 데이터 포인트의 레이블을 확인할 수 있습니다.

```python
print(km.labels_)
print(np.unique(km.labels_, return_counts=True))
```

- `km.labels_`는 각 데이터가 속한 클러스터 레이블을 제공합니다.
- `np.unique(km.labels_, return_counts=True)`는 클러스터별 데이터 개수를 출력합니다.


## 클러스터 결과 시각화

각 클러스터에 속한 이미지를 시각화하여 클러스터링 결과를 확인합니다.

### 이미지 출력 함수 정의

```python
import matplotlib.pyplot as plt

def draw_fruits(arr, ratio=1):
    n = len(arr)
    rows = int(np.ceil(n / 10))
    cols = n if rows < 2 else 10
    fig, axs = plt.subplots(rows, cols,
                            figsize=(cols * ratio, rows * ratio), squeeze=False)
    for i in range(rows):
        for j in range(cols):
            if i * 10 + j < n:
                axs[i, j].imshow(arr[i * 10 + j], cmap='gray_r')
            axs[i, j].axis('off')
    plt.show()
```

### 클러스터별 이미지 출력

```python
draw_fruits(fruits[km.labels_ == 0])
draw_fruits(fruits[km.labels_ == 1])
draw_fruits(fruits[km.labels_ == 2])
```

### 클러스터 중심 시각화

```python
draw_fruits(km.cluster_centers_.reshape(-1, 100, 100), ratio=3)
```

- 클러스터 중심은 각 그룹의 대표 이미지를 나타냅니다.


## 새로운 데이터 예측

K-Means 모델을 사용하여 새로운 데이터 포인트가 각 클러스터와 얼마나 가까운지 확인할 수 있습니다.

```python
print(km.transform(fruits_2d[100:101]))
print(km.predict(fruits_2d[100:101]))
draw_fruits(fruits[100:101])
```

- `km.transform`은 클러스터 중심까지의 거리를 반환합니다.
- `km.predict`는 해당 데이터가 속한 클러스터를 예측합니다.


## 최적 클러스터 수 찾기

클러스터 수(k)를 조정하여 inertia 값을 분석합니다. inertia는 클러스터 내 데이터가 중심에서 얼마나 가까운지를 측정합니다.

```python
inertia = []
for k in range(2, 7):
    km = KMeans(n_clusters=k, n_init='auto', random_state=42)
    km.fit(fruits_2d)
    inertia.append(km.inertia_)

plt.plot(range(2, 7), inertia)
plt.xlabel('k')
plt.ylabel('inertia')
plt.show()
```

- `inertia` 값이 급격히 감소하는 지점에서 최적의 k를 선택할 수 있습니다.


## 요약

K-Means 클러스터링은 이미지 데이터를 그룹화하는 데 유용한 도구입니다. 본 실습에서는 과일 이미지를 사용하여 다음을 수행했습니다:

1. 데이터를 클러스터링하여 그룹화.
2. 클러스터 결과를 시각화.
3. 새로운 데이터를 예측.
4. 최적의 클러스터 수를 찾기 위한 분석.

K-Means를 이미지 데이터에 적용하면 데이터 패턴을 효과적으로 이해할 수 있습니다.

