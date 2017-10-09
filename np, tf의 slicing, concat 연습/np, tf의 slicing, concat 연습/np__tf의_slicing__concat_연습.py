import numpy as np

A=np.zeros((3,3,2), dtype=float) #np.array((3,3,2), dtype=float)으로 하면 shape가 터지는 걸 보아 초기화를 해줘야 할 듯 하다.
print(A.shape)
A[2,2,1]=1
print(A)
a=np.array([[1,2,3,4],[2,3,4,5],[3,4,5,6],[4,5,6,7]])
b=a[:2, 1:3]
print(b)
b[0][0]=77
print(a[0][1])

row1=a[1, :]
row2=a[1:2, :]

print(row1.shape)
print(row2.shape)

print(row1)
print(row2)

coloum1 = a[:, 1]
coloum2 = a[:, 1:2]

print(coloum1.shape)
print(coloum2.shape)

print(coloum1)
print(coloum2)

D = np.zeros((3,3,3))
for i in range(3):
    for j in range(3):
        for k in range(3):
            D[i,j,k]=100*i+10*j+k

print(D)

rowD1=D[0:1,0:1,:]

print(rowD1.shape)
print(rowD1)

rowD2=D[0,0,:]
print(rowD2.shape)
print(rowD2)

coloumD1=D[0:1, :, 0:1]
print(coloumD1.shape)
print(coloumD1)

coloumD2=D[0, :, 0]
print(coloumD2.shape)
print(coloumD2)

heightD1=D[:, 0:1, 0:1]
print(heightD1.shape)
print(heightD1)

heightD2=D[:, 0, 0]
print(heightD2.shape)
print(heightD2)#호오... 신기하군.. python의 세계에 온걸 환영한다 느낌?

x_data=np.array(
    [[1],[2.3],[3.5],[2.4],[1.6],[7.3],[-3.4],[-4.6]])
v_data=np.array(
    [[2.3],[1.5],[-2.3],[2.3],[-1],[1],[-3.2],[2.3]])
des_data=np.array(
    [[-1.3],[4.5],[-3.2],[1.3],[-1.5],[-1],[-1.2],[5.3]])

A = np.concatenate([x_data, v_data, des_data],1)
print(A)
print(A.shape)