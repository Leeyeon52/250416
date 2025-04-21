import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
warnings.filterwarnings(action='ignore')

# XOR 문제를 위한 데이터 생성

def main():

    training_data = np.array([[0,0],[0,1],[1,0],[1,1]], 'float32')
    target_data = np.array([[0],[1],[1],[0]], 'float32')

    '''
    1. 다층 퍼셉트론 모델을 만듭니다.
    '''

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.summary()
    '''
    2. 모델 학습 방법을 설정합니다.
    '''

    model.compile(loss='mse',  optimizer='adam',  metrics=['binary_accuracy'])

    '''
    3. 모델을 학습시킵니다. epochs를 자유롭게 설정해보세요.
    ''' 

    hist = model.fit(training_data, target_data, epochs = 20)
    print(type(hist))

    score = hist.history['binary_accuracy'][-1]
    
    print('최종 정확도: ', score*100, '%')
    
    return hist
            
if __name__ == "__main__":
    main()