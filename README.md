# OSW-final


## • Explain what you do in your project
tumor data set이랑 sklearn 라이브러리를 이용하여 4가지의 tumor를 분류하는 project입니다.

우리가 할 일은 사용할 model 찾기와 적절한 파라미터 조절입니다 추가적으로 저는 train data 증강까지 하였습니다.
## Train data set 설명
1. **Dataset Structure:**
     - `glioma_tumor`
     - `meningioma_tumor`
     - `no_tumor`
     - `pituitary_tumor`
2. **Image Loading:**
   데이터 로딩 후 사이즈를 64X64로 resize해 주었다.

   이후 gray색으로 바꾸고 np.array()로 바꾸어 이미지 데이터를 바꿔 준 후 데이터를 나누었다

3. **Data Augmentation:**
   데이터 일반화를 시키기위해 가지고 있는 Train data를 이미지 증강하여 Train data set에 추가해 주었습니다.

   먼저 픽셀의 대비를 조절 한 후 명암을 조절한 데이터 set을 추가하였고 이후 원래 train data에서 noisy를 낀 데이터를 추가했습니다.
   또한 Train data의 수를 늘리기 위해 train,test split을 0.00001을 사용하였습니다.
## Models

- Support Vector Machine (SVM)
- k-Nearest Neighbors (KNN)
- Extra Trees
- Histogram-based Gradient Boosting


## • Explain the algorithm you choose

  
1.SVC(Support Vector Classification)>>데이터를 가장 잘 나누는 결정 경계를 찾는 알고리즘으로 고차원 데이터(이미지 데이터)일 수록 유리하기 때문에 사용하였습니다.

2. KNN(K-Nearest Neighbors)>>K개의 가까운 데이터를 보고 결정하는 알고리즘으로 과적합 가능성이 낮기 때문에 사용하였습니다.

3.ExtraTree>>Random Forest와 비슷하나 ExtraTree는 트리를 만들 때 랜덤하게 특성을 선택하고 데이터를 샘플링해 과적합 가능성이 낮아 사용하였습니다.
 
4.HistGradientBoosting>>히스토그램 기반해서 Gradient Boosting 알고리즘을 구현한 모델입니다. 이 모델 또한 과적합 가능성이 낮아 사용하였습니다.

 ## • Explain hyper-parameter of the function

 
 1.SVC>>kernel(데이터를 특징 공간으로 매핑하는 역할을 하고, 'rbf'를 사용하였습니다.), C(제약 조건의 강도,C=6로 조정하여 작은 제약조건을 사용하였습니다.), gamma(가까운 값의 가중치 정도로 0.01를 사용하였습니다) 해당 파라미터 값들은 grid_search알고리즘을 돌려 찾아냈습니다.

2.KNN(K-Nearest Neighbors)>>n_neighbors(예측 할 때 고려할 이웃의 수로 3을 사용하였습니다.), n_jobs(사용할 CPU 관련 파라미터로 별 의미 없습니다.), P(거리는 유클리드 거리를 사용했습니다.),weights(예측에 사용되는 가중치로 distance 가까운 이웃에 더 큰 가중치가 적용되게 하였습니다.), algorithm(이웃을 검색할 때 사용하는 알고리즘으로 ball_tree를 사용하였습니다.) 이는 grid Search를 통해 알아낸 파라미터입니다.

3.ExtraTree>>하이퍼 파라미터 조정 안했습니다.

4.HistGradientBoostingClassifier>>max_iter(반복 횟수로 높을 수록 과적합의 가능성이 커집니다. 50으로 설정했습니다.)max_depth(트리의 최대 깊이로 높을 수록 과적합의 가능성이 커집니다. 5로 설정했습니다)  learning_rate(학습률로 각 트리의 기여 정도 매개변수로 과적합을 줄이려고 0.2를 사용했습니다.)








