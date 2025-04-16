import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class LinearRegressionModel:
    def __init__(self):
        # 가중치와 Bias를 변수로 초기화합니다.
        self.W = tf.Variable(tf.random.normal([1]), name='weight')
        self.b = tf.Variable(tf.zeros([1]), name='bias')

    def __call__(self, x):
        # 선형 회귀 모델의 예측값을 반환합니다.
        return self.W * x + self.b

    def loss(self, y_predicted, y_true):
        # MSE (Mean Squared Error) 손실 함수를 계산합니다.
        return tf.reduce_mean(tf.square(y_predicted - y_true))

def train(model, inputs, outputs, learning_rate):
    with tf.GradientTape() as t:
        current_loss = model.loss(model(inputs), outputs)
    # 손실 함수에 대한 가중치 W와 Bias b의 기울기를 계산합니다.
    dW, db = t.gradient(current_loss, [model.W, model.b])
    # 계산된 기울기를 사용하여 가중치와 Bias를 업데이트합니다.
    model.W.assign_sub(learning_rate * dW)
    model.b.assign_sub(learning_rate * db)

def main():
    # 학습 데이터 설정
    X = tf.constant([1.0, 2.0, 3.0, 4.0], shape=(4, 1))
    y = tf.constant([2.0, 4.0, 6.0, 8.0], shape=(4, 1))

    # 선형 회귀 모델 인스턴스 생성
    linear_model = LinearRegressionModel()

    # 학습률 설정
    learning_rate = 0.1
    epochs = 100

    print("초기 loss =", linear_model.loss(linear_model(X), y).numpy())
    print("초기 W =", linear_model.W.numpy(), "초기 b =", linear_model.b.numpy())

    # 학습 시작
    for epoch in range(epochs):
        train(linear_model, X, y, learning_rate)
        if (epoch + 1) % 10 == 0:
            loss = linear_model.loss(linear_model(X), y).numpy()
            print("Epoch {:03d}: Loss = {:.3f}, W = {}, b = {}".format(epoch + 1, loss, linear_model.W.numpy(), linear_model.b.numpy()))

    print("\n최종 loss =", linear_model.loss(linear_model(X), y).numpy())
    print("최종 W =", linear_model.W.numpy(), "최종 b =", linear_model.b.numpy())

if __name__ == "__main__":
    main()