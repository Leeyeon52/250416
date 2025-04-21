import tensorflow as tf
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(100)
tf.random.set_seed(100)

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

def Visualize(x_data, y_data, predictions):
    # 데이터 출력
    plt.scatter(x_data, y_data)
    plt.savefig('reg2_data.png')
    plt.show()

    # 곡선형 분포 데이터와 예측값 출력
    plt.scatter(x_data, predictions, color='red')
    plt.savefig('reg2_prediction.png')
    plt.show()

def main():
    # 데이터 생성
    x_data = np.linspace(0, 10, 100)
    y_data = 1.5 * x_data**2 -12 * x_data + np.random.randn(*x_data.shape)*2 + 0.5
    
    # 신경망 모델 생성
    model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(20, input_dim=1, activation='relu'), # if x<=0: x=0
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dense(1)
    ])

    model.summary()
    
    # 최적화 모델 설정
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    # 학습 시작 
    history = model.fit(x_data, y_data, epochs=500, verbose=1)
    
    # 학습된 모델을 사용하여 예측값 생성 및 저장
    predictions = model.predict(x_data)
    
    Visualize(x_data, y_data, predictions)
    
    return history, model
    
if __name__ == '__main__':
    main()