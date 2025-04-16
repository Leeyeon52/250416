import numpy as np
import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


'''
Tensorflow 1.x
'''

def tf1():
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

    # 상수
    a = tf.constant(5)
    b = tf.constant(3)

    # 계산 정의
    add_op = a + b

    # 세션 시작
    with tf.Session() as sess:
        result_tf1 = sess.run(add_op)
        a_np = sess.run(a)  # Evaluate 'a' to get its NumPy value
        b_np = sess.run(b)  # Evaluate 'b' to get its NumPy value

    return a_np, b_np, result_tf1

'''
Tensorflow 2.0
'''

def tf2():
    import tensorflow as tf
    # Ensure no tf.compat.v1 and tf.disable_v2_behavior() here

    # 상수
    a = tf.constant(5)
    b = tf.constant(3)

    # 즉시 실행 연산
    result_tf2 = tf.add(a,b)

    return a.numpy(), b.numpy(), result_tf2.numpy()

def main():
    tf1_a, tf1_b, tf1_result = tf1()
    tf2_a, tf2_b, tf2_result = tf2()

    print('Tensorflow 1.x 결과:')
    print('  a:', tf1_a)
    print('  b:', tf1_b)
    print('  result_tf1:', tf1_result)
    print('\nTensorflow 2.0 결과:')
    print('  a:', tf2_a)
    print('  b:', tf2_b)
    print('  result_tf2:', tf2_result)

if __name__ == "__main__":
    main()