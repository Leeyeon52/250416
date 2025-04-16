import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def getParameters(X, y):
    f = len(X[0])           # Step01: feature 수
    w = [1.0] * f           # Step01: 초기 가중치 float로 설정

    while True:
        wPrime = [0.0] * f  # Step02: 변화량 초기화

        for i in range(len(y)):
            r = 0
            for j in range(f):  # Step03: 선형 조합 계산
                r += X[i][j] * w[j]
            v = sigmoid(r)      # Step03: 시그모이드

            for j in range(f):  # Step03: gradient 누적
                wPrime[j] += -((v - y[i]) * v * (1 - v) * X[i][j])

        # Step04: 종료 조건 - 변화량이 작으면 반복 종료
        if all(abs(wp) < 0.001 for wp in wPrime):
            break

        for j in range(f):  # Step04: 가중치 업데이트
            w[j] += wPrime[j]

    return w

def main():
    '''
    이 코드는 수정하지 마세요.
    '''
    X = [(1, 0, 0), (1, 0, 1), (0, 0, 1)]
    y = [0, 1, 1]

    '''
    # 아래의 예제 또한 테스트 해보세요.
    X = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
    y = [0, 0, 1, 1, 1, 1, 1, 1]

    # 아래의 예제를 perceptron이 100% training할 수 있는지도 확인해봅니다.
    X = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
    y = [0, 0, 0, 1, 0, 1, 1, 1]
    '''

    print(getParameters(X, y))

if __name__ == "__main__":
    main()
