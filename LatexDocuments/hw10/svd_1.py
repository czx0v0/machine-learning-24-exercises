import numpy as np
from scipy.linalg import null_space

# 初始化A矩阵
A = np.array([[2,4],[1,3],[0,0],[0,0]],dtype=float)
m,n = A.shape

# 计算A的转置乘A，并求特征值、特征向量
ATA = np.array(A.T@A)
ATA_value,ATA_vector = np.linalg.eig(ATA)

# 对特征值按降序排序
argsort_value_index = np.argsort(-ATA_value)
ATA_sort_value = ATA_value[argsort_value_index]
# 按特征值降序排序特征向量作为V矩阵列向量并将其转置的VT矩阵
VT = ATA_vector[argsort_value_index]
print(f'V^T:\n{VT}')
# 对特征值取根号获取奇异值
sigma_list = []
for i in range (0,ATA_sort_value.shape[0]):
    value = np.sqrt(ATA_sort_value[i])
    sigma_list.append(value)
print(f'奇异值:\n{sigma_list}')

# 构造m*n对角矩阵sigma_matrix
sigma_matrix = np.zeros((m,n))
sigma_matrix[:len(sigma_list),:len(sigma_list)] = np.diag(sigma_list)
print(f'对角矩阵:\n{sigma_matrix}')

# 计算U矩阵
U = np.zeros((m,m)) # 初始化U矩阵
for i in range(m):
    # 计算U1
    if i < len(sigma_list) and (sigma_list[i]>0):
        # i<=r时 计算u_j = 1/sigma_j*A*v_j
        uj = 1/sigma_list[i] * (A@ VT[i])
        U[i] = uj
    # 计算U2
    else:
         # i>r时 计算U2为A^T的零空间的标准正交基
        U2 = null_space(A.T).T
        for j in range(len(U2)):
            U[i] = U2[j]
        break
# 转置得到U矩阵
U= U.T
print(f'U:\n{U}')

A_res = U@sigma_matrix@VT
# 验证结果
print(f'U * Sigma * V^T :\n{A_res}')
print(f'A:\n{A}')
