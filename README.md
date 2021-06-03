# DACON - 태양광 발전량 예측 AI 경진대회



## 시계열ㅣPinball Loss | LGBM

![대회요약](https://user-images.githubusercontent.com/28820900/120660100-f1d60780-c4c1-11eb-9a87-dcac0571b5d8.PNG)

### https://dacon.io/competitions/official/235680/overview/description

- ## 데이터

  - train.csv : 훈련용 데이터 (1개 파일)

    - 3년(Day 0~ Day1094) 동안의 기상 데이터, 발전량(TARGET) 데이터 

    

  - test.csv : 정답용 데이터 (81개 파일)

    - 2년 동안의 기상 데이터, 발전량(TARGET) 데이터 제공 

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



## 데이터 전처리

- 피쳐 재생산

- ```python
  'theta', 'Hour_bef', 'IsRain', 'Zenith', 'Elevation','GHI', 'Season','Aggr','Daytime','RH_bef','Cos_hour','Target_hour_mean'
  ```

  





## 모델

- LGBM 모델 사용

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

  
