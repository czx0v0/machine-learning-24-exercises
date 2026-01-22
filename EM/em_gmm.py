import numpy as np
import math
Y = np.array([-67, -48, 6, 8, 14, 16, 23, 24, 28, 29, 41, 49, 56, 60, 75])
# 辅助函数 高斯分布密度
def cal_gaussian_phi_k(y, sigma2_k, mu_k):
    res = 1 / (math.sqrt(2 * math.pi * sigma2_k))
    res = res * math.exp(-((y - mu_k) * (y - mu_k)) / (2 * sigma2_k))
    return res
# em估计gmm参数
def gmm_em(Y, alpha_1, sigma2_1, sigma2_2, mu_1, mu_2, iter_max=5):
    print(f'初值：alpha_1 = {alpha_1},sigma2_1 = {sigma2_1},sigma2_2 = {sigma2_2},mu_1 = {mu_1},mu_2 = {mu_2}')
    N = len(Y)
    # 参数迭代值
    alpha_1i = alpha_1
    sigma2_1i = sigma2_1  # 是sigma^2
    sigma2_2i = sigma2_2
    mu_1i = mu_1
    mu_2i = mu_2
    # 更新参数时便于计算j1和j2
    gamma_j1_N = np.ones(N)
    gamma_j2_N = np.ones(N)
    gamma_j2_yj_N = np.ones(N)
    gamma_j1_yj_N = np.ones(N)
    gamma_yj_mu1_2_N = np.ones(N)
    gamma_yj_mu2_2_N = np.ones(N)
    for i in range(iter_max):
        # E步
        for j in range(N):
            # 循环计算gamma_jk
            yj = Y[j]
            gamma_j1_N[j] = (alpha_1i * cal_gaussian_phi_k(yj, sigma2_1i, mu_1i))/((alpha_1i * cal_gaussian_phi_k(yj, sigma2_1i, mu_1i)) +
                             (1 - alpha_1i) * cal_gaussian_phi_k(yj,sigma2_2i,mu_2i))
            gamma_j2_N[j] = ((1 - alpha_1i) * cal_gaussian_phi_k(yj, sigma2_2i, mu_2i))/((alpha_1i * cal_gaussian_phi_k(yj, sigma2_1i, mu_1i)) +
                             (1 - alpha_1i) * cal_gaussian_phi_k(yj,sigma2_2i,mu_2i))
            gamma_j1_yj_N[j] = yj * gamma_j1_N[j]
            gamma_j2_yj_N[j] = yj * gamma_j2_N[j]
            gamma_yj_mu1_2_N[j] = gamma_j1_N[j] * ((yj - mu_1i) ** 2)
            gamma_yj_mu2_2_N[j] = gamma_j1_N[j] * ((yj - mu_2i) ** 2)
        # M步 更新参数
        mu_1i = sum(gamma_j1_yj_N) / sum(gamma_j1_N)
        mu_2i = sum(gamma_j2_yj_N) / sum(gamma_j2_N)
        sigma2_1i = sum(gamma_yj_mu1_2_N) / sum(gamma_j1_N)
        sigma2_2i = sum(gamma_yj_mu2_2_N) / sum(gamma_j2_N)
        alpha_1i = sum(gamma_j1_N) / N
        # 打印
        print(f'第{i + 1}轮：mu_1i = {mu_1i},mu_2i = {mu_2i},sigma2_1i = {sigma2_1i},sigma2_2i = {sigma2_2i},alpha_1i = {alpha_1i}')
    return alpha_1i, sigma2_1i, sigma2_2i, mu_1i, mu_2i
alpha_1i, sigma2_1i, sigma2_2i, mu_1i, mu_2i = gmm_em(Y, 0.5, 1000, 1000, 100, 100, 100)
print(f'mu_1i = {mu_1i},mu_2i = {mu_2i},\sigma2_1i = {sigma2_1i},sigma2_2i = {sigma2_2i},alpha_1i = {alpha_1i}')
print('----------')
alpha_1i, sigma2_1i, sigma2_2i, mu_1i, mu_2i = gmm_em(Y, 0.4, 100, 100, 10, 10, 1000)
print(f'mu_1i = {mu_1i},mu_2i = {mu_2i},sigma2_1i = {sigma2_1i},sigma2_2i = {sigma2_2i},alpha_1i = {alpha_1i}')
