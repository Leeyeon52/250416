import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

np.random.seed(100)

class LinearModel:
    def __init__(self):
        self.W = tf.Variable(tf.random.normal([1]), dtype=tf.float32) # 랜덤 초기화
        self.b = tf.Variable(tf.random.normal([1]), dtype=tf.float32) # 랜덤 초기화

    def __call__(self, X):
        return self.W * X + self.b

def loss(y, pred):
    return tf.reduce_mean(tf.square(pred - y))

def main():
    # 데이터 생성
    x_data = np.linspace(0, 10, 50).reshape(-1, 1).astype(np.float32) # 데이터 타입 명시
    y_data = (4 * x_data + np.random.randn(*x_data.shape) * 4 + 3).astype(np.float32) # 데이터 타입 명시

    linear_model = LinearModel()
    epochs = 200
    learning_rate = 0.01 # 학습률 조정
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate) # Adam 옵티마이저 사용

    for epoch_count in range(epochs):
        with tf.GradientTape() as tape:
            y_pred_data = linear_model(x_data)
            real_loss = loss(y_data, y_pred_data)

        gradients = tape.gradient(real_loss, [linear_model.W, linear_model.b])
        optimizer.apply_gradients(zip(gradients, [linear_model.W, linear_model.b]))

        if (epoch_count % 20 == 0):
            print(f"Epoch count {epoch_count}: Loss value: {real_loss.numpy()}")
            print('W: {}, b: {}'.format(linear_model.W.numpy(), linear_model.b.numpy()))

            plt.figure()
            plt.scatter(x_data, y_data)
            plt.plot(x_data, y_pred_data, color='red')
            plt.savefig(f'prediction_adam_epoch_{epoch_count}.png')
            plt.show()

if __name__ == "__main__":
    main()