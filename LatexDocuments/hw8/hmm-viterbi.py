import numpy as np

# 输入数据
# lambda=（A，B，pi）
pi = np.array([0.2, 0.4, 0.4])
# N=3
A = np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
B = np.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]])
# T = 4， 红0白1
O = np.array([0, 1, 0, 1])  # 长度为T的观测序列O
Q = np.array([1, 2, 3])  # 所有可能的状态集合Q

def viterbi(A, B, pi, O, Q):
    N = len(Q)  # 所有可能的状态共有N个
    T = len(O)  # 观测序列的长度为T
    delta_t = np.zeros([T, N])  # 存储delta
    Psi_t = np.zeros([T, N])  # 存储Psi
    I_star = np.array([0, 0, 0, 0])  # 存储最优路径序列
    print("开始递推")
    # 开始递推
    for t in range(T):
        print(f't={t + 1}')
        # 初始化 t=1
        if t == 0:
            for i in range(N):
                delta_t[0][i] = pi[i] * B[i][O[0]]
                Psi_t[0][i] = 0
                print(f'delta_1[{i + 1}]:{delta_t[0][i]:.6f},Psi_1[{i + 1}]:{Psi_t[0][i]}')
        # t=2,3,...,T
        else:
            for i in range(N):
                delta_t[t][i] = np.max([np.max(delta_t[t - 1][j - 1] * A[j - 1][i]) * B[i][O[t]] for j in Q])
                delta_t_A = [delta_t[t - 1][j - 1] * A[j - 1][i] for j in Q]
                Psi_t[t][i] = np.argmax(delta_t_A) + 1
                print(f'delta_{t + 1}[{i + 1}]:{delta_t[t][i]:.6f},Psi_{t + 1}[{i + 1}]:{Psi_t[t][i]}')
    # 终止 t=T
    print('终止')
    P_star = np.max(delta_t[T - 1])
    I_star[T - 1] = int(np.argmax(delta_t[T - 1])) + 1

    # 回溯
    print('开始回溯')
    for i in range(T - 1):
        t = T - i - 2
        # 对t=T-1，T-2，...,1
        I_star[t] = int(Psi_t[t + 1][I_star[t + 1] - 1])
    print("结束")
    return I_star

if __name__ == '__main__':
    I_star = viterbi(A, B, pi, O, Q)
    print(f'最优序列为：{I_star}')
