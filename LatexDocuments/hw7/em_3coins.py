import numpy as np

Y = np.array([1,1,0,1,0,0,1,0,1,1])
def cal_em_3coins(Y,pi_0 =0.46 ,p_0= 0.55,q_0= 0.67,iter_max = 3):
    # 初值
    pi_i= pi_0
    p_i= p_0
    q_i= q_0
    # 迭代步骤
    n = len(Y)
    # 存储j方便计算
    mu = np.zeros(n) # mu_j
    y_times_mu = np.zeros(n) # y_j*mu_j
    y_times_1_mu = np.zeros(n) # y_j*(1-mu_j)
    mu_1 = np.zeros(n) # (1-mu_j)

    for i in range(iter_max):
        # E步 计算mu_j
        for j in range(n):
            mu[j] = (pi_i*p_i**Y[j] * (1-p_i)**(1-Y[j]))/(pi_i*p_i**Y[j] * (1-p_i)**(1-Y[j])+(1-pi_i)*q_i**Y[j] * (1-q_i)**(1-Y[j]))
            y_times_mu[j] = mu[j]*Y[j]
            y_times_1_mu[j] = Y[j]*(1-mu[j])
            mu_1[j] = 1-mu[j]
        # M步 更新参数
        pi = (1/n)*sum(mu)
        p_i = sum(y_times_mu)/sum(mu)
        q_i = sum(y_times_1_mu)/sum(mu_1)

        print(f'第{i+1}轮迭代，y_j = 1,mu={mu[0]}.y_j=0,mu={mu[2]}')
        print(f'pi_{i+1}={pi_i},p_{i+1}={p_i},q_{i+1}= {q_i}')
    return pi_i, p_i, q_i

print('-----')
pi,p,q = cal_em_3coins(Y,0.5,0.5,0.5,2)
print('初值：pi= 0.5,p=0.5,q= 0.5')
print(f'pi= {pi},p={p},q= {q}')

print('-----')
pi,p,q = cal_em_3coins(Y,0.4,0.6,0.7,2)
print('初值：pi= 0.4,p=0.6,q= 0.7')
print(f'pi= {pi},p={p},q= {q}')

print('-----')
pi,p,q = cal_em_3coins(Y,0.46,0.55,0.67,2)
print('初值：pi= 0.4,p=0.6,q= 0.7')
print(f'pi= {pi},p={p},q= {q}')
print('-----')