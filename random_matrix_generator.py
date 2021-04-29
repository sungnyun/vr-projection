import numpy as np
import random
import math
import time

def generate_random_one_minus_one_matrix_with_seed(row_d, ratio=0.1,s=1):
    k=int(row_d*ratio)
    scale_val = math.sqrt(s/k)
    random_vec = np.zeros(row_d * k)
    for i in range(row_d * k):
        random_vec[i] = random.choice([- scale_val,  scale_val])
    return np.reshape(random_vec, (row_d, k))
 
#이 함수의 output은 2차원 tensor에 곱해져서 기존 2차원 tensor의 한 dimension의 값을 줄임으로서 기존 정보를 ratio배만큼 압축해주는 함수입니다. 기존 정보가 (row_d,  col_d)의 tensor일 때 (row_d*ratio, col_d) matrix가 함수의 output matrix와 곱해짐으로서 얻어집니다. 함수의 output matrix가 생성되는 과정은 다음과 같습니다. row_d와 ratio를 곱한 값이 k일 때, scale_val=math.sqrt(1/k)인데 이 out matrix의 element는 {scale_val, -scale_val} 중 한 값을 1/2의 확률로 각각 갖게 되게끔 형성합니다.


def generate_random_ternary_matrix_with_seed(row_d, ratio=0.1,s=1.5):
    k=int(row_d*ratio)
    scale_val = math.sqrt(s/k)
    random_vec = np.zeros(row_d * k)
    for i in range(row_d * k):
        random_vec[i] = random.choice([- scale_val, 0, scale_val])
    return np.reshape(random_vec, (row_d, k))
'''
def general_generate_random_ternary_matrix_with_seed(row_d, ratio=0.1,s=1):
    k=int(row_d*ratio)
    scale_val = math.sqrt(s/k)
    val_lst = [scale_val, -scale_val] + [0] * (2*int(s) - 2)
    random_vec = np.array([random.choice(val_lst) for _ in range(row_d * k)])
    return np.reshape(random_vec, (row_d, k))
'''

def general_generate_random_ternary_matrix_with_seed(row_d, ratio=0.1, s=1):
    k = int(row_d*ratio)
    scale_val = math.sqrt(s/k)
    p = [1/(2*s), 1-1/s, 1/(2*s)]
    val_lst = [1, 0, -1]
    random_vec = scale_val * np.random.choice(val_lst, size=(row_d, k), p=p) 
    return random_vec
    
