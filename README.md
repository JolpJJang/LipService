# LipService
딥러닝 기반 입술 모양 인식하여 구어 텍스트화

```DataCollect_lip&face localization.ipynb```: 한국 Data를 수집하는 코드(lip과 face)  
```cropping.ipynb``` : 얼굴과 입술 부분만 crop 시켜서 저장하는 코드  
```Preprocessing_lip.ipynb```: 폴더내 파일들을 정리하여 모델의 input으로 넣기 위한 전처리 단계


----

```models.py```  : CNN_LSTM & deep_CNN_LSTM & pretrained VGG + LSTM & fine_tuned VGG + LSTM 모델 저장
```TRAIN.py``` : Train을 위한 코드
