from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# import elice_utils
# elice_utils = elice_utils.EliceUtils()

np.random.seed(100)
tf.random.set_seed(100) # Use the modern TensorFlow 2.x way

'''
1. 다층 퍼셉트론 분류 모델을 만들고, 학습 방법을 설정해
    학습시킨 모델을 반환하는 MLP 함수를 구현하세요.

    Step01. 다층 퍼셉트론 분류 모델을 생성합니다.
            여러 층의 레이어를 쌓아 모델을 구성해보세요.

    Step02. 모델의 손실 함수, 최적화 방법, 평가 방법을 설정합니다.

    Step03. 모델을 학습시킵니다. epochs를 자유롭게 설정해보세요.
'''

def MLP(x_train, y_train):

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)), # 첫 번째 레이어 (Input layer)
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax') # 마지막 레이어 (Output layer)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=20, verbose=0)

    return model

def main():

    try:
        x_train = np.loadtxt('train_images.csv', delimiter =',', dtype = np.float32)
        y_train = np.loadtxt('train_labels.csv', delimiter =',', dtype = np.float32)
        x_test = np.loadtxt('test_images.csv', delimiter =',', dtype = np.float32)
        y_test = np.loadtxt('test_labels.csv', delimiter =',', dtype = np.float32)
    except FileNotFoundError as e:
        print(f"오류: 데이터 파일({e.filename})을 찾을 수 없습니다. './data/' 디렉토리에 파일이 있는지 확인해주세요.")
        return
    except Exception as e:
        print(f"데이터 로딩 중 오류 발생: {e}")
        return

    # 이미지 데이터를 0~1범위의 값으로 바꾸어 줍니다.
    x_train, x_test = x_train / 255.0, x_test / 255.0

    try:
        model = MLP(x_train,y_train)

        # 학습한 모델을 test 데이터를 활용하여 평가합니다.
        loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

        print('\nTEST 정확도 :', test_acc)

        # 임의의 3가지 test data의 이미지와 레이블값을 출력하고 예측된 레이블값 출력
        predictions = model.predict(x_test)
        rand_n = np.random.randint(0, len(x_test), size=3) # IndexError 방지

        for i in rand_n:
            img = x_test[i].reshape(28,28)
            plt.imshow(img,cmap="gray")
            plt.title(f"Label: {int(y_test[i])}, Prediction: {np.argmax(predictions[i])}") # 이미지 제목 추가
            plt.show()
            plt.savefig(f"test_image_{i}.png") # 로컬에 이미지 저장 (선택 사항)
            # elice_utils.send_image("test.png") # elice_utils 대신 주석 처리

            print("Label: ", y_test[i])
            print("Prediction: ", np.argmax(predictions[i]))

    except Exception as e:
        print(f"모델 학습 또는 평가 중 오류 발생: {e}")

if __name__ == "__main__":
    main()
    