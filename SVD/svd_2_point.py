import numpy as np

A= [[0,20,5,0,0],[10,0,0,3,0],[0,0,0,0,1],[0,0,1,0,0]]
np.set_printoptions(precision=6,suppress=True)
U,sigma_matrix,V = np.linalg.svd(A)
print(f'U:\n{U}')
print(f'Sigma:\n{sigma_matrix}')
print(f'V:\n{V.T}')