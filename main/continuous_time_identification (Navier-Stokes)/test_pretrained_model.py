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
# Training Data
idx = np.random.choice(N * T, N_train, replace=False)
x_train = x[idx, :]
y_train = y[idx, :]
t_train = t[idx, :]
u_train = u[idx, :]
v_train = v[idx, :]
p_train = p[idx, :]

# Training
model = PhysicsInformedNN(x_train, y_train, t_train, p_train, u_train, v_train, layers)
model.load_model('./pinn_ns_results/pinn_ns_model_20000.ckpt')
x_, y_, t_, p_, u_, v_ = my_utils_1.load_data()
x_, y_, t_ = x_.reshape(-1, 1), y_.reshape(-1, 1), t_.reshape(-1, 1)
p_, u_, v_ = p_.reshape(-1, 1), u_.reshape(-1, 1), v_.reshape(-1, 1)
u_pred, v_pred, p_pred = model.predict(x_, y_, t_)
x_, y_, t_ = x_.flatten(), y_.flatten(), t_.flatten()
p_, u_, v_ = p_.flatten(), u_.flatten(), v_.flatten()
p_field, p_field_pred, u_norm, u_norm_pred = my_utils_1.transform_data(x_, y_, t_, p_, u_, v_, p_pred, u_pred, v_pred)
my_utils_1.update_plot(p_field, p_field_pred, u_norm, u_norm_pred, 0, plot=True)
plt.show()
