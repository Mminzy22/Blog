---
layout: post
title: "Machine Learning 04: 데이터 이해와 전처리"
date: 2024-12-05
categories: [Machine Learning] 
---


머신러닝의 성공은 **데이터**에 달려 있다고 해도 과언이 아닙니다. 데이터를 어떻게 준비하고 구성하느냐에 따라 모델의 성능과 신뢰도가 결정됩니다. 이번 글에서는 데이터의 역할과 중요성을 이해하고, 데이터셋의 종류와 품질 관리의 필요성을 살펴보겠습니다.


#### 데이터의 역할

**데이터란?**  
데이터는 머신러닝 모델이 학습하고 예측을 수행하기 위한 핵심 자원입니다.  
- **입력 데이터(Input):** 모델에 제공되는 데이터.  
- **출력 데이터(Output):** 모델이 예측하거나 생성해야 하는 결과.

**데이터의 중요성**  
1. **학습과 일반화:**  
   머신러닝 모델은 데이터를 통해 학습하며, 학습 데이터가 다양하고 대표성을 띠어야 새로운 데이터에 대해 일반화된 예측을 할 수 있습니다.
2. **모델 성능 향상:**  
   고품질의 데이터는 학습 속도를 높이고 모델의 성능을 극대화합니다.
3. **결과의 신뢰성:**  
   데이터가 불완전하거나 왜곡되면 모델 결과도 왜곡될 수 있습니다.


#### 데이터셋의 종류와 구성

**1. 데이터셋의 종류**  
머신러닝에 사용되는 데이터셋은 학습 과정과 평가 단계에서 역할에 따라 나뉩니다:

- **학습 데이터셋 (Training Set):**  
  모델이 학습하는 데 사용되는 데이터.  
  - 입력 데이터와 정답(레이블)으로 구성.  
  - 예: 입력 이미지와 해당 클래스(고양이, 개 등).

- **검증 데이터셋 (Validation Set):**  
  모델의 하이퍼파라미터 튜닝 및 성능을 검증하는 데 사용.  
  - 학습 데이터와 겹치지 않아야 함.  
  - 예: 학습 중 과적합(overfitting)을 방지.

- **테스트 데이터셋 (Test Set):**  
  최종 모델의 성능을 평가하기 위한 데이터.  
  - 학습 및 검증 데이터와 독립적이어야 함.  
  - 예: 새로운 환경에서 모델의 일반화 능력 확인.

**2. 데이터 구성 요소**  
- **특징(Feature):**  
  모델이 학습할 수 있는 데이터의 속성.  
  - 예: 주택 가격 예측에서 크기, 위치, 방 개수 등이 특징에 해당.
- **레이블(Label):**  
  모델이 예측해야 하는 목표 값.  
  - 예: 고양이 사진인지 개 사진인지 분류.
- **샘플(Sample):**  
  데이터셋의 한 개의 단위.  
  - 예: 하나의 이미지, 하나의 행.


#### 데이터 품질의 중요성

**왜 데이터 품질이 중요한가?**  
머신러닝 모델은 데이터의 품질에 따라 성능이 크게 좌우됩니다.  
품질이 낮은 데이터는 모델에 잘못된 패턴을 학습시키거나 일반화 성능을 떨어뜨릴 수 있습니다.

**1. 데이터 품질 문제**
- **결측값(Missing Values):**  
  데이터셋에 값이 누락되어 있는 경우.  
  - 예: 설문조사에서 특정 질문에 답변하지 않은 경우.
- **이상값(Outliers):**  
  데이터의 일반적인 범위를 벗어난 값.  
  - 예: 주택 가격 예측에서 터무니없는 가격 데이터.
- **중복 데이터(Duplicates):**  
  동일한 데이터가 반복되어 나타나는 경우.  
  - 예: 데이터셋에 같은 고객 기록이 여러 번 포함된 경우.

**2. 데이터 품질 관리 방법**
- **데이터 정제(Data Cleaning):**  
  결측값 대체, 이상값 처리, 중복 제거 등을 통해 데이터 품질을 개선.
- **데이터 정규화 및 표준화:**  
  스케일 차이를 줄여 모델 학습을 효율적으로 수행.
- **대표성 있는 데이터 확보:**  
  충분한 데이터 샘플과 다양한 케이스를 포함하여 학습 데이터의 편향을 방지.


#### 정리

데이터는 머신러닝의 가장 중요한 자원으로, 데이터의 역할과 품질은 모델의 성능을 좌우합니다.  
- **데이터셋의 구성**은 학습, 검증, 테스트 단계로 나뉘며, 각 단계에 맞는 데이터를 사용해야 합니다.
- **데이터 품질 관리**는 결측값 처리, 이상값 제거 등을 포함하며, 이는 신뢰성 있는 결과를 보장합니다.

> **다음 글 예고:**  
> 데이터 전처리의 실제 기법인 **"결측값 처리"** 방법에 대해 알아보겠습니다!