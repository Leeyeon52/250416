import numpy as np

# 사용할 1차 선형 회귀 모델
def linear_model(w0, w1, X):
    f_x = w0 + w1 * X
    return f_x

# MSE 손실 함수 정의
def Loss(f_x, y):
    ls = np.mean((f_x - y)**2)
    return ls

# gradient 계산 함수
def gradient_descent(w0, w1, X, y):
    f_x = linear_model(w0, w1, X)  # ✅ f_x를 여기서 계산
    n = len(y)
    gradient0 = np.sum(f_x - y) / n
    gradient1 = np.sum((f_x - y) * X) / n
    return np.array([gradient0, gradient1])

# 메인 학습 루프
def main():
    X = np.array([1, 2, 3, 4]).reshape((-1, 1))
    y = np.array([3.1, 4.9, 7.2, 8.9]).reshape((-1, 1))

    # 파라미터 초기화
    w0 = 0
    w1 = 0

    # 학습률
    lr = 0.001

    # 반복 학습
    for i in range(1000):
        f_x = linear_model(w0, w1, X)
        gd = gradient_descent(w0, w1, X, y)

        # 가중치 업데이트
        w0 = w0 - lr * gd[0]
        w1 = w1 - lr * gd[1]

        # 100회마다 출력
        if i % 100 == 0:
            loss = Loss(linear_model(w0, w1, X), y)
            print(f"{i}번째 loss : {loss}")
            print(f"{i}번째 w0, w1 : {w0}, {w1}\n")

    return w0, w1

if __name__ == '__main__':
    main()
