# -*- coding: utf-8 -*-
import numpy as np
import scipy.sparse.linalg as sla
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Loading data
data=loadmat("Data164.mat")

# Reading measurement matrix
A=data['A']

# Plotting sparsity pattern for 10 random projections
plt.figure(figsize=(16,7))

N = 2
N2 = 5
for i in range(0,N*5):  
    plt.subplot(N,N2,i+1)
    index = np.random.randint(0,A.shape[0])
    proj = A[index,:].reshape(164,164)
    plt.spy(proj,marker='.')

plt.show()

# Reading sinogram data
m=data['m']

sinogram = data['m']

# Plotting sinogram data as an image
plt.figure(figsize=(6,10))
plt.imshow(sinogram,cmap='Greys',aspect=0.5)
plt.show()

# Reshaping sinogram data into a 1D array
b_e = sinogram.reshape([164*120,1],order='F')

# calculating pseudoinverse
k = 200
P,S,QT = sla.svds(A,k=k)

A_k_plus = QT.T @ np.diag(1/S) @ P.T

# solving for ct result
X = A_k_plus @ b_e

# plotting ct result
plt.figure(figsize=(8,8))
plt.imshow(X.reshape(164,164).T,cmap='gray')
plt.show()