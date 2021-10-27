# DACON - 태양광 발전량 예측 AI 경진대회



## 시계열ㅣPinball Loss | LGBM

![대회요약](https://user-images.githubusercontent.com/28820900/120660100-f1d60780-c4c1-11eb-9a87-dcac0571b5d8.PNG)

### https://dacon.io/competitions/official/235680/overview/description

- ## 데이터

  - train.csv : 훈련용 데이터 (1개 파일)

    - 3년(Day 0~ Day1094) 동안의 기상 데이터, 발전량(TARGET) 데이터 

    

  - test.csv : 정답용 데이터 (81개 파일)

    - 2년 동안의 기상 데이터, 발전량(TARGET) 데이터 제공 

- 7일(Day 0~ Day6) 동안의 데이터를 인풋으로 활용하여, 향후 2일(Day7 ~ Day8) 동안의 30분 간격의 발전량(TARGET)을 예측

  ```python
  train = pd.read_csv('./data/train/train.csv')
  train
  ```

![train_data](https://user-images.githubusercontent.com/28820900/120660116-f7335200-c4c1-11eb-91b3-f305289764c4.png)

  - Hour - 시간
  - Minute - 분
  - DHI - 수평면 산란일사량(Diffuse Horizontal Irradiance (W/m2))
  - DNI - 직달일사량(Direct Normal Irradiance (W/m2))
  - WS - 풍속(Wind Speed (m/s))
  - RH - 상대습도(Relative Humidity (%))
  - T - 기온(Temperature (Degree C))
  - Target - 태양광 발전량 (kW)



## 관련 논문 분석

- 차왕철, 숭실대학교, 2015, 태양광발전에 영향을 미치는 요소 분석을 통한 연간 발전량 예측에 관한 연구

  ![논문_1](https://user-images.githubusercontent.com/28820900/120928447-a4ef6c80-c71f-11eb-80ef-c48be859ff7c.PNG)
  
  - GHI 와 천정각 등 태양 고도를 결정짓는 요소들과 기상 상태를 추정할 수 있는 주요 기상 요소들을 이용하여 태양광 발전량에 대한 예측률을 높일 수 있었다고 설명.

  

## 피쳐 재생산

```python
'theta', 'target_hour_bef', 'IsRain', 'Zenith', 'Elevation','GHI' ,'Aggr','Daytime','RH_hour_bef','Cos_hour','Target_hour_mean'
```

#### GHI = DHI + (DNI X Cosθz)

- θz : 천정각 (zenith angle)

- DNI와 DHI를 합한 GHI가 있습니다.

- 단일 변수로 당일 발전량(TARGET)을 설명할 때 DNI, DHI보다 GHI가 설명력이 높으며, GHI 파생변수들이 설명력이 좋음.

#### target_hour_bef / RH_hour_bef : 30분 전의 Target, RH 값, 시계열 모델을 사용하지 않기에 직전 시간의 Target 정보를 모델에 포함시켜주는 것이 효과적

#### Zenith : 천정각

#### Target_hour_mean : 같은 시간 발전량(TARGET)의 평균

#### isRain :  비가 오는지 여부. 습도 데이터를 사용하여 생성.

#### DayTime : 해당 날짜에 해가 떠있는 시간을 계산, Grouping & Aggregation 하여 평균과 표준편차를 구하여 피쳐 추가에 사용.




## 데이터 시각화

- Daytime 별 발전량(TARGET) 분포

![daytime별 data 분포](https://user-images.githubusercontent.com/28820900/120928432-943ef680-c71f-11eb-9b80-1cb646a8aa4b.PNG)

  - 각 index 는 test data 의 각 행을 의미하며 각 행은 7일치의 데이터로 다른 행들과 시간적으로 독립적, 
  - Daytime 을 이용하여 이러한 index 들을 효과적으로 분류해낼 수 있는 지표라 판단. (Grouping 에 효과적)

## 모델

- LGBM 모델 사용
  - 제안 모델에서는 시계열의 특성을 고려하지 않고, 태양 고도를 결정짓는 요소들과 기상 상태를 추정할 수 있는 주요 기상 요소들을 고려하여 신경망 모델을 이용하여 단기 예측을 수행 (Day6 -> Day7, Day8 예측)

- ```python
  # Get the model and the predictions in (a) - (b)
  def LGBM(q, X_train, Y_train, X_valid, Y_valid, X_test):
      
      # (a) Modeling  
      model = LGBMRegressor( objective='quantile', alpha=q,
                           n_estimators=10000, bagging_fraction=0.7, learning_rate=0.027)                   
                           
      model.fit(X_train, Y_train, eval_metric = ['quantile'], 
            eval_set=[(X_valid, Y_valid)], early_stopping_rounds=300, verbose=0)
      
      #fig,ax = plt.subplots(figsize=(10,6))
      #plot_importance(model,ax=ax)
  
      # (b) Predictions
      pred = pd.Series(model.predict(X_test).round(2))
      return pred, model
  ```

  
