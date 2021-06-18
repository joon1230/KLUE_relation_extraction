# Pstage_2_KLUE_Relation_extraction



## Model 관련 코드는 다음과 같이 구성 되어있습니다.

__training__ 
- train.py  
    ko-electra 모델을 구현한 train 코드입니다.

- train_xlm.py  
    xlm_roberta_large 모델을 구현한 train 코드입니다.

__inference__
- inference.py

__preprocessing data, data loader__
- load_data.py

__Train DEMO__
```
python train.py
```

### 각 디렉토리는 모델을 구현하였습니다.  
( 안의 파일 구조는 위와 같습니다.)
### Reduce_sentence
- xlm_reduce_size  
    data loader에서 entity에 초점을 맞춰 문장의 길이를 줄인 방법을 활용한 모델을 구현하였습니다.  

    __Train DEMO__
    ```
    python xlm_reduce_size/Train.py
    ```

- xlm_binary  
    이진 분류 모델을 구현 하였습니다.   

    __Train DEMO__
    ```
    python xlm_binary/Train.py
    ```

<br></br>
### Ensemble 관련 코드

- Ensemble.ipynb  
    앙상블 실험을 진행한 노트북 파일입니다.

- submission_report.ipynb  
    기존의 모델들의 추론 결과 분포를 시각화한 노트북 파일입니다.


<br></br>
### EDA ( Explore Data Analysis ) 관련 코드

- EDA.ipynb  
    단순한 EDA를 해본 노트북 파일입니다.

- baseline_review.ipynb  
    baseline code를 뜯어본 노트북 파일입니다.