import numpy as np
from numpy import linalg as LA
import math
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D

np.set_printoptions(
    suppress=True,
    precision=4
)

def matricaKamere(org,imgs):
    n = len(imgs)
    X = []
 
    for i in range(n):
        A, B, C, D = imgs[i]
        a, b, d = org[i]
        
        X.append([0, 0, 0, 0, -d*A, -d*B, -d*C, -d*D, b*A, b*B, b*C, b*D])
        X.append([d*A, d*B, d*C, d*D, 0, 0, 0, 0, -a*A, -a*B, -a*C, -a*D])
        
    X = np.array(X)
    
    _, _, V = LA.svd(X)
    T = V[-1].reshape(3, 4)
 
    T = T / T[-1, -1]
    T = np.where(np.isclose(T, 0) , 0.0 , T)
    return T

def kalibracijaKamere(T):
    T0 = np.delete(T, 3, 1)
    T0i = LA.inv(T0)

    T1 = T0i[:, 0]
    T2 = T0i[:, 1]
    T3 = T0i[:, 2]

    Q1 = T1
    Q2 = T2 - ((np.dot(T2, Q1) / np.dot(Q1, Q1)) * Q1)
    Q3 = T3 - ((np.dot(T3, Q2) / np.dot(Q2, Q2)) * Q2) - ((np.dot(T3, Q1) / np.dot(Q1, Q1)) * Q1)

    Q1 = Q1 / LA.norm(Q1)
    Q2 = Q2 / LA.norm(Q2)
    Q3 = Q3 / LA.norm(Q3)

    K = np.dot(T0, np.column_stack([Q1, Q2, Q3]))
    K = K / K[-1, -1]
    K = np.where(np.isclose(K, 0), 0.0, K)

    return K

def centar(T):
    C1 = LA.det(np.delete(T,0,1))
    C2 = LA.det(np.delete(T,1,1))
    C3 = LA.det(np.delete(T,2,1))
    C4 = LA.det(np.delete(T,3,1))

    C = np.array([C1, -C2, C3, -C4]) / -C4
    C = np.where(np.isclose(C, 0) , 0.0 , C)

    return C

def kameraA(T):
    T0 = np.delete(T,3,1)
    T0i = LA.inv(T0)
    
    T1 = T0i[:, 0]
    T2 = T0i[:, 1]
    T3 = T0i[:, 2]
    
    Q1 = T1 
    Q2 = T2 - ((np.dot(T2, Q1) / np.dot(Q1, Q1)) * Q1)
    Q3 = T3 - ((np.dot(T3, Q2) / np.dot(Q2, Q2)) * Q2) - ((np.dot(T3, Q1) / np.dot(Q1, Q1)) * Q1)

    Q1 = Q1 / LA.norm(Q1)
    Q2 = Q2 / LA.norm(Q2)
    Q3 = Q3 / LA.norm(Q3)
    
    A = np.column_stack([Q1, Q2, Q3])
    
    if LA.det(A) < 0:
        A *= -1
    
    A = np.where(np.isclose(A, 0) , 0.0 , A)
    return A

org1 =  np.array([1,-1,-1])*(np.array([2016,0,0]) -  np.array([[1003, 336, 1], [1441, 501, 1], [1032, 743, 1], [579, 497, 1], [656, 968, 1], [1014, 1254, 1], [1349, 963, 1], [1006, 767, 1]] ))


imgs1 = np.array([[0, 0, 3, 1], [0, 3, 3, 1], [3, 3, 3, 1], [3, 0, 3, 1], [3, 0, 0, 1], [3, 3, 0, 1], [0, 3, 0, 1], [2, 2, 3, 1]])

T1 = matricaKamere(org1,imgs1)
print("Matrica kamere:\n", T1)
print()
print("Matrica kalibracije kamere:\n", kalibracijaKamere(T1))
print()
print("Pozicija centra kamere: ", centar(T1))
print()

print("Spoljasnja matrica kamere: \n", kameraA(T1))
print()

tacke = np.array([[3, 0, 0], [3, 3, 0], [0, 3, 0], [0, 0, 0], [3, 0, 3], [3, 3, 3], [0, 3, 3], [0, 0, 3]])

ivice = np.array([[tacke[0], tacke[1], tacke[2], tacke[3]], 
         [tacke[4], tacke[5], tacke[6], tacke[7]],  
         [tacke[0], tacke[1], tacke[5], tacke[4]],  
         [tacke[2], tacke[3], tacke[7], tacke[6]], 
         [tacke[1], tacke[2], tacke[6], tacke[5]], 
         [tacke[0], tacke[3], tacke[7], tacke[4]]])


b_x = np.array([10, 0, 0])
b_y = np.array([0, 10, 0])
b_z = np.array([0, 0, 10])

A = kameraA(T1)
K = kalibracijaKamere(T1)
C = centar(T1)
C = C[:3]

x_kamere = A[0]
y_kamere = A[1]
z_kamere = A[2]



fig = plt.figure()
axs = fig.add_subplot(111, projection='3d')
axs.add_collection3d(Poly3DCollection(ivice, facecolors='teal', linewidths=0.2, edgecolors='black', alpha=1))

i = [0, 0, 0]

axs.quiver(*i, *b_x, color = 'blue')
axs.quiver(*i, *b_y, color = 'green')
axs.quiver(*i, *b_z, color = 'red')


axs.quiver(*C, *x_kamere, color = 'blue')
axs.quiver(*C, *y_kamere, color = 'green')
axs.quiver(*C, *z_kamere, color = 'red')


axs.set_xlabel('X axis')
axs.set_ylabel('Y axis')
axs.set_zlabel('Z axis')
axs.set_xlim([-1, 9])
axs.set_ylim([-1, 9])
axs.set_zlim([0, 9])

plt.show()