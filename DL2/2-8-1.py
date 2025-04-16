import tensorflow as tf
import numpy as np
# from visual import * # visual 모듈은 제공되지 않아 주석 처리했습니다.

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

np.random.seed(100)
tf.random.set_seed(100)

def main():

    # XOR 게이트 입출력 데이터
    x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

    '''
    1. 다층 퍼셉트론 모델을 만듭니다.
    '''

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(16, input_dim=2, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    '''
    2. 모델 학습 방법을 설정합니다.
    '''

    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['binary_accuracy'])

    '''
    3. 모델을 학습시킵니다.
    '''

    history = model.fit(x_data, y_data, epochs=5000, verbose=0) # epochs를 늘려 정확도 향상

    '''
    4. 학습된 모델을 사용하여 예측값 생성 및 저장
    '''

    predictions = model.predict(x_data)
    print("예측값:\n", predictions)
    print("\n정확도:", history.history['binary_accuracy'][-1])

    # Visualize(x_data, y_data, predictions) # visual 모듈이 없어 주석 처리

    return history, model

if __name__ == '__main__':
    main()