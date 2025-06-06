import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''
1. 두 실수와 연산 종류를 입력받는 함수입니다. 코드를 살펴보세요.
'''

def insert():

    x = float(input('정수 또는 실수를 입력하세요. x : '))
    y = float(input('정수 또는 실수를 입력하세요. y : '))
    cal = input('어떤 연산을 할것인지 입력하세요. (+, -, *, /)')

    return x, y, cal

'''
2. 입력받는 연산의 종류 cal에 따라 연산을 수행하고
    결과를 반환하는 calcul() 함수를 완성하세요.
'''

def calcul(x,y,cal):

    x_tensor = tf.constant(x)
    y_tensor = tf.constant(y)
    result = tf.constant(0.0)

    # 더하기
    if cal == '+':
        result = tf.add(x_tensor, y_tensor)

    # 빼기
    elif cal == '-':
        result = tf.subtract(x_tensor, y_tensor)

    # 곱하기
    elif cal == '*':
        result = tf.multiply(x_tensor, y_tensor)

    # 나누기
    elif cal == '/':
        # 0으로 나누는 경우를 방지
        if y != 0:
            result = tf.divide(x_tensor, y_tensor)
        else:
            return "Error: 0으로 나눌 수 없습니다."

    else:
        return "Error: 지원하지 않는 연산자입니다."

    return result.numpy()

'''
3. 두 실수와 연산 종류를 입력받는 insert 함수를 호출합니다. 그 다음
    calcul 함수를 호출해 실수 사칙연산을 수행하고 결과를 출력합니다.
'''

def main():

    x, y, cal = insert()

    print(calcul(x,y,cal))

if __name__ == "__main__":
    main()