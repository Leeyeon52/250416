import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    
    # XOR 문제를 위한 데이터 생성
    
    training_data = np.array([[0,0],[0,1],[1,0],[1,1]], "float32")
    target_data = np.array([[0],[1],[1],[0]], "float32")
    
    '''
    1. 다층 퍼셉트론 모델을 생성합니다. (레이어 및 뉴런 수 증가)
    '''
    
    model = tf.keras.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(2,))) # 더 많은 뉴런
    model.add(layers.Dense(64, activation='relu')) # 추가적인 은닉층
    model.add(layers.Dense(1, activation='sigmoid')) # 출력층

    '''
    2. 모델의 손실 함수, 최적화 방법, 평가 방법을 설정합니다.
    '''
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['binary_accuracy'])

    '''
    3. 모델을 학습시킵니다. epochs를 늘려봅니다.
    ''' 
    
    hist = model.fit(training_data, target_data, epochs=500, validation_split=0.2, verbose=0) # epochs 증가
    
    _, accuracy = model.evaluate(training_data, target_data, verbose=0)
    print('최종 정확도: ', accuracy*100, '%')
    return hist

if __name__ == "__main__":
    main()