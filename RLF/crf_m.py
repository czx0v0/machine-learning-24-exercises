import numpy as np

# y可能的取值为1，2
# 用Y存储所有可能的观测序列y
Y= np.array([[1,1,1],[1,1,2],[1,2,1],[1,2,2],[2,1,1],[2,1,2],[2,2,1],[2,2,2]])
# 矩阵M1，M2，M3，M4
M1 = np.array([[0,0],[0.5,0.5]])
M2 = np.array([[0.3,0.7],[0.7,0.3]])
M3 = np.array([[0.5,0.5],[0.6,0.4]])
M4 = np.array([[0,1],[0,1]])
# start = 2,stop = 2
start = 2
stop = 2
def crf_matrix(Y,M1,M2,M3,M4,start,stop):
    # 计算规范化因子
    Z =  (M1@M2@M3@M4)[start-1][stop-1]
    print(f'Z={Z}')
    # 用于存储概率最大的状态序列
    max_y = [1,1,1]
    max_P = 0
    # 对每个可能的序列，计算规范化概率
    for y in Y:
        # 对每个可能的观测序列
        print(f'y={y},',end = '')
        P_unnormalized= M1[2-1][y[0]-1]*M2[y[0]-1][y[1]-1]*M3[y[1]-1][y[2]-1]*M4[y[2]-1][2-1]
        print(f'非规范化概率：{P_unnormalized:.3f},',end='' )
        P_normalized= P_unnormalized/Z
        print(f'规范化概率：{P_normalized:.3f}.')
        if P_normalized>max_P:
            max_P=P_normalized
            max_y=y
    print(f'概率最大的状态序列为：{max_y},最大概率为：{max_P}')
    return max_y,max_P
if __name__ == '__main__':
    max_y,max_P = crf_matrix(Y,M1,M2,M3,M4,start,stop)
