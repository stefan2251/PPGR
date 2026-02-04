import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def equation(left, right):
    return [a * b for a in left for b in right]

def equations(T1, T2, m1, m2):
    return np.array([
        m1[1]*T1[2] - m1[2]*T1[1],
        -m1[0]*T1[2] + m1[2]*T1[0],
        m2[1]*T2[2] - m2[2]*T2[1],
        -m2[0]*T2[2] + m2[2]*T2[0]
    ])

p1l = [118,316]
p2l = [659,106]
p3l = [987,303]
p4l = [405,686]
p5l = [227,573]
p6l = [662,314]
p7l = [888,542]
p8l = [450,886]

p9l = [331,384]
p10l = [461,136]
p11l = [630,139]
p12l = [581,407]
p13l = [356,470]
p14l = [452,175]
p15l = [628,218]
p16l = [580,490]

p1d = [572,153]
p2d = [911,389]
p3d = [472,591]
p4d = [239,248]
p5d = [587,355] 
p6d = [853,624]
p7d = [512,822]
p8d = [302,466]

p9d = [386,156]
p10d = [694,185]
p11d = [661,263]
p12d = [314,223]
p13d = [399,214]
p14d = [693,255]
p15d = [659, 341]
p16d = [332, 296]


left = [p1l, p2l, p3l, p4l, p5l, p6l, p7l, p8l, p9l, p10l, p11l, p12l, p13l, p14l, p15l, p16l]
right = [p1d, p2d, p3d, p4d, p5d, p6d, p7d, p8d, p9d, p10d, p11d, p12d, p13d, p14d, p15d, p16d]

left = [np.array([1200 - x[0], x[1], 1]) for x in left]
right = [np.array([1200 - x[0], x[1], 1]) for x in right]


matrica = [equation(l, r) for l, r in zip(left, right)]


U, D, V = LA.svd(matrica)
F = np.array(V[-1])
F = F.reshape(3, 3).T
print("Fundamentalna matrica:\n", F)
print()
print("Determinanta F = ", LA.det(F))
print()

U, D, V = LA.svd(F)
e1 = np.array(V[-1]) 
e1 = e1 * (1 / e1[-1])

e2 = np.array(U.T[-1]) 
e2 = e2 * (1 / e2[-1]) 

print("e1: ", e1)
print("e2: ", e2)
print()

D = np.diag(D)
D1 = np.diag([1, 1, 0])
D1 = D1 @ D
F1 = (U @ D1) @ V
print("Determinanta F1 = ", LA.det(F1))
print()

K1 = np.array(
    [[1200, 0, 600],
     [0, 1200, 450],
     [0, 0, 1]])


E = (K1.T @ F1) @ K1

print("E matrica:\n", E)
print()

Q0 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
E0 = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])

U, SS, V = np.linalg.svd(-E)

EC = (U @ E0) @ U.T
AA = (U @ Q0.T) @ V

print("Matrica EC: \n", EC)
print()
print("Matrica A: \n", AA)  
print()


C = [EC[2, 1], EC[0, 2], EC[1, 0]]
T2 = np.hstack((K1, np.zeros((K1.shape[0], 1))))
C1 = -AA.T @ C

T1 = np.vstack(((K1 @ AA.T).T, K1 @ C1)).T
print("Matrica prve kamere T1:\n", T1)
print()
print("Matrica druge kamere T2:\n", T2)
print()



koor3D = [] 
print("3D koordinate:\n")
for m1, m2 in zip(left, right):
    sistem = equations(T1, T2, m1, m2)
    _, _, V = LA.svd(sistem)
    tmp = np.array(V[-1])
    tmp = tmp[:-1] / tmp[-1]
    tmp = tmp * 10
    koor3D.append(tmp)
    print(tmp)


ivice1 = [
    [koor3D[0], koor3D[1], koor3D[2], koor3D[3]],
    [koor3D[4], koor3D[5], koor3D[6], koor3D[7]],
    [koor3D[0], koor3D[1], koor3D[5], koor3D[4]],
    [koor3D[3], koor3D[2], koor3D[6], koor3D[7]],
    [koor3D[1], koor3D[2], koor3D[6], koor3D[5]],
    [koor3D[0], koor3D[3], koor3D[7], koor3D[4]]
]


ivice2 = [
    [koor3D[8], koor3D[9], koor3D[10], koor3D[11]],
    [koor3D[12], koor3D[13], koor3D[14], koor3D[15]],
    [koor3D[8], koor3D[9], koor3D[13], koor3D[12]],
    [koor3D[11], koor3D[10], koor3D[14], koor3D[15]],
    [koor3D[9], koor3D[10], koor3D[14], koor3D[13]],
    [koor3D[8], koor3D[11], koor3D[15], koor3D[12]]
]

x_c = AA.T[0]
y_c = AA.T[1]
z_c = AA.T[2]

figure = plt.figure()
axes = figure.add_subplot(111, projection='3d')
axes.add_collection3d(Poly3DCollection(ivice1, facecolors='blue', linewidths=0.5, edgecolors='black', alpha=0.4))
axes.add_collection3d(Poly3DCollection(ivice2, facecolors='red', linewidths=0.5, edgecolors='black', alpha=0.4))

axes.quiver(*C, *x_c, color = 'blue')
axes.quiver(*C, *y_c, color = 'green')
axes.quiver(*C, *z_c, color = 'red')


axes.set_xlabel('X osa')
axes.set_ylabel('Y osa')
axes.set_zlabel('Z osa')

axes.set_xlim([-4, 4])
axes.set_ylim([-4, 4])
axes.set_zlim([-10, 1])

plt.show()