import numpy as np
import pandas as pd
import random

def loss_function(rui,u,f):
    m,n = rui.shape
    K = u.shape[1]
    loss = 0
    for i in range(m):
        for j in range(n):
            if rui[i,j] > 0:
                pred = 0
                reg = 0
                for k in range(K):
                    pred += u[i,k] * f[k,j]
                    # reg += 5 * (u[i,k] * u[i,k] + f[k,j] * f[k,j])
                loss += (rui[i,j] - pred) * (rui[i,j] - pred)
    #loss1 = np.sum((rui - u.dot(f) * (rui > 0))**2 + 5*((u**2).dot(np.ones_like(f)) + np.ones_like(u).dot(f ** 2) )*(rui>0)**2)
    #print(loss1-loss)
    return loss / np.sum(rui > 0)

def grad(rui,u,f):
    m,n = rui.shape
    K = u.shape[1]
    for i in range(m):
        for j in range(n):
            if rui[i,j] > 0:
                e = rui[i,j]
                for k in range(K):
                    e -= u[i,k] * f[k,j]
                for k in range(K):
                    u[i,k] += 2 * 0.002 * (e * f[k,j] - 0.02 * u[i,k])
                    f[k,j] += 2 * 0.002 * (e * u[i,k] - 0.02 * f[k,j])
    return u,f

def als(rui,u,f):
    m,n = rui.shape
    K = u.shape[1]
    k_eye = np.eye(K)
    for mm in range(m):
        f_sub = f[:,(rui[mm, :] > 0)]
        rui_sub = rui[mm, (rui[mm, :] > 0)].reshape(-1, 1)
        u[mm:mm+1,:] = np.linalg.inv(f_sub.dot(f_sub.T)+5*k_eye).dot(f_sub).dot(rui_sub).T
    for nn in range(n):
        u_sub = u[(rui[:,nn] > 0),:]
        rui_sub = rui[(rui[:, nn] > 0),nn].reshape(-1, 1)
        f[:,nn:nn+1] = np.linalg.inv(u_sub.T.dot(u_sub)+5*k_eye).dot(u_sub.T).dot(rui_sub)
    return u,f

features = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('u.data',sep = '\t',names = features)

n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]

from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(df, test_size=0.25)

# data = np.zeros((n_users,n_items))
# for line0 in train_data.itertuples():
#     data[line0[1]-1,line0[2]-1] = line0[3]
# da = {
#     'predict': data
# }
# pd.DataFrame(da).to_csv('data.csv', index=False, encoding="utf8")
# np.savetxt("data.txt", np.array(data, dtype=int), encoding="utf8")


rui = np.zeros((n_users,n_items))
for line in train_data.itertuples():
    rui[line[1]-1,line[2]-1] = line[3]

validation = np.zeros((n_users,n_items))
for line1 in test_data.itertuples():
    validation[line1[1]-1,line1[2]-1] = line1[3]


m,n = rui.shape
k = 10
u = np.random.random((m,k))
f = np.random.random((k,n))

k1 = 20
u1 = np.random.random((m,k1))
f1 = np.random.random((k1,n))

validation_loss_K10 = []
validation_loss_K20 = []
for iter in range(15):
    #i = random.randint(0,m)
    #sample = rui[i:i+1,:]
    #u,f = grad(sample,u,f)
    u,f = als(rui,u,f)
    loss = loss_function(validation,u,f)
    validation_loss_K10.append(loss)

    u1,f1 = als(rui,u1,f1)
    loss = loss_function(validation,u1,f1)
    validation_loss_K20.append((loss))

# predict = u.dot(f)
#
# pred = {
#     'predict': predict
# }
# np.savetxt("predict.txt", np.array(predict + 0.5, dtype=int), encoding="utf8")
# pd.DataFrame(pred).to_csv('predict.csv', index = False, encoding="utf8")

import matplotlib.pyplot as plt

plt.figure(figsize=(18, 6))
plt.plot(validation_loss_K10, "-", color="b", label="K=10")
plt.plot(validation_loss_K20, "-", color="r", label="K=20")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.title("The graph of absolute diff value varing with the number of iterations")

plt.show()
