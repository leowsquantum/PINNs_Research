import os
# Suppress TensorFlow logging (1: INFO, 2: WARNING, 3: ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import warnings
warnings.filterwarnings("ignore")
import my_utils_1
import numpy as np
import matplotlib.pyplot as plt
from NavierStokes import PhysicsInformedNN
import scipy.io

import matplotlib
matplotlib.rcParams['text.usetex'] = False

N_train = 5000  # 5000

layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 2]

# Load Data
data = scipy.io.loadmat('../Data/cylinder_nektar_wake.mat')

U_star = data['U_star']  # N x 2 x T
P_star = data['p_star']  # N x T
t_star = data['t']  # T x 1
X_star = data['X_star']  # N x 2

N = X_star.shape[0]
T = t_star.shape[0]

# Rearrange Data
XX = np.tile(X_star[:, 0:1], (1, T))  # N x T
YY = np.tile(X_star[:, 1:2], (1, T))  # N x T
TT = np.tile(t_star, (1, N)).T  # N x T

UU = U_star[:, 0, :]  # N x T
VV = U_star[:, 1, :]  # N x T
PP = P_star  # N x T

x = XX.flatten()[:, None]  # NT x 1
y = YY.flatten()[:, None]  # NT x 1
t = TT.flatten()[:, None]  # NT x 1

u = UU.flatten()[:, None]  # NT x 1
v = VV.flatten()[:, None]  # NT x 1
p = PP.flatten()[:, None]  # NT x 1

######################################################################
######################## Noiseles Data ###############################
######################################################################

idx = np.random.choice(N * T, N_train, replace=False)
x_train = x[idx, :]
y_train = y[idx, :]
t_train = t[idx, :]
u_train = u[idx, :]
v_train = v[idx, :]
p_train = p[idx, :]

def get_model(model_path):
    model = PhysicsInformedNN(x_train, y_train, t_train, p_train, u_train, v_train, layers)
    model.load_model(model_path)
    return model


def get_data():
    x_, y_, t_ = x_train.flatten(), y_train.flatten(), t_train.flatten()
    p_, u_, v_ = p_train.flatten(), u_train.flatten(), v_train.flatten()
    return x_, y_, t_, p_, u_, v_


ns_samples = [5000]
models = []
for n in ns_samples:
    print('n_sample: ', n)
    model_path = f'./pinn_ns_results/n_sample_{n}/pinn_ns_model_20000.ckpt'
    models.append(get_model(model_path))
x_, y_, t_, p_, u_, v_ = my_utils_1.load_data(50)
x_, y_, t_ = x_.reshape(-1, 1), y_.reshape(-1, 1), t_.reshape(-1, 1)
p_, u_, v_ = p_.reshape(-1, 1), u_.reshape(-1, 1), v_.reshape(-1, 1)
uvp_preds = np.array([model.predict(x_, y_, t_) for model in models])  # (n_sample, 3, x*y, 1)
uvp_preds = np.squeeze(uvp_preds, axis=-1)  # (n_sample, 3, x*y)
u_preds, v_preds, p_preds = uvp_preds[:, 0, :], uvp_preds[:, 1, :], uvp_preds[:, 2, :]
x_, y_, t_, p_, u_, v_ = x_.flatten(), y_.flatten(), t_.flatten(), p_.flatten(), u_.flatten(), v_.flatten()
fig, ax = plt.subplots(2, len(ns_samples) + 1, figsize=(20, 10))
p_field = []
u_norm = []
for i in range(len(ns_samples)):
    p_field, p_field_pred, u_norm, u_norm_pred = my_utils_1.transform_data(x_, y_, t_, p_, u_, v_, p_preds[i], u_preds[i], v_preds[i])
    ax[0, i].imshow(p_field_pred)
    ax[1, i].imshow(u_norm_pred)
ax[0, len(ns_samples)].imshow(p_field)
ax[1, len(ns_samples)].imshow(u_norm)
plt.show()
