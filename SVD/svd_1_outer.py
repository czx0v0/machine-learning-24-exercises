# 外积展开式
# A = sigma_1u_1v_1^T+sigma_2u2v_2^T
A_outer = sigma_list[0]*np.outer(U[:,0],VT[0,:])+sigma_list[1]*np.outer(U[:,1],VT[1,:])
print(f'A外积展开式计算结果:\n{A_outer}')
