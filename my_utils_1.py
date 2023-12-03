from matplotlib import animation

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import time

plt.rcParams['text.usetex'] = False
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
def load_orig_data():
    # load data
    data_path = '../Data/cylinder_nektar_wake.mat'
    data = scipy.io.loadmat(data_path)
    # keys: ['__header__', '__version__', '__globals__', 'X_star', 't', 'U_star', 'p_star']
    r = np.transpose(data['X_star'])
    t_orig = np.array(data['t']).flatten()
    x = np.tile(r[0], len(t_orig))
    y = np.tile(r[1], len(t_orig))
    t = np.repeat(t_orig, len(r[0]))
    u = np.array(data['U_star']).transpose((1, 2, 0))
    p_orig = np.transpose(data['p_star'])
    ux_orig = u[0]
    uy_orig = u[1]
    # print('x_orig:', r[0].shape, 'y_orig:', r[1].shape,  't_orig:', t_orig.shape, 'p_orig:', p_orig.shape, 'ux_orig:', ux_orig.shape, 'uy_orig:', uy_orig.shape)

    ux = ux_orig.flatten()
    uy = uy_orig.flatten()
    p = p_orig.flatten()


    # print('x:', x.shape, 'y:', y.shape, 't:', t.shape, 'p:', p.shape, 'ux:', ux.shape, 'uy:', uy.shape)

    return x, y, t, p, ux, uy, r, t_orig, p_orig, ux_orig, uy_orig

x_, y_, t_, p_, ux_, uy_, r_, t_orig_, p_orig_, ux_orig_, uy_orig_ = load_orig_data()

def load_data(ti=0):
    # slice data at t=0
    x = x_[ti * len(r_[0]):(ti + 1) * len(r_[0])]
    y = y_[ti * len(r_[0]):(ti + 1) * len(r_[0])]
    t = t_[ti * len(r_[0]):(ti + 1) * len(r_[0])]
    p = p_[ti * len(r_[0]):(ti + 1) * len(r_[0])]
    ux = ux_[ti * len(r_[0]):(ti + 1) * len(r_[0])]
    uy = uy_[ti * len(r_[0]):(ti + 1) * len(r_[0])]

    return x, y, t, p, ux, uy

# def load_model_params():
#     model_path = 'result_5000_0/pinn-large-tanh-bs128-lr0.005-lrstep1-lrgamma0.8-epoch20/ckpt-39/ckpt.pt'
#     model_param = torch.load(model_path)
#     return model_param
#
#
# def load_model(model_param):
#     model = Pinn(1, 8)
#     model.load_state_dict(model_param)
#     model.eval()
#     return model


def transform_data(x, y, t, p, ux, uy, p_pred, ux_pred, uy_pred):
    # convert pressure to scalar field
    x_axis = sorted(set(x.tolist()))
    y_axis = sorted(set(y.tolist()))
    x_min = x_axis[0]
    x_max = x_axis[-1]
    y_min = y_axis[0]
    y_max = y_axis[-1]
    x_step = (x_max - x_min) / (len(x_axis) - 0.5)
    y_step = (y_max - y_min) / (len(y_axis) - 0.5)
    p_field = np.zeros((len(y_axis), len(x_axis)), float)
    p_field_pred = np.zeros((len(y_axis), len(x_axis)), float)

    for i in range(len(x)):
        xi = int((x[i] - x_min) / x_step)
        yi = int((y[i] - y_min) / y_step)
        p_field_pred[yi][xi] = p_pred[i]

    for i in range(len(x)):
        xi = int((x[i] - x_min) / x_step)
        yi = int((y[i] - y_min) / y_step)
        p_field[yi][xi] = p[i]

    u_norm = np.zeros((len(y_axis), len(x_axis)), float)
    ux_np = ux
    uy_np = uy
    for i in range(len(x)):
        xi = int((x[i] - x_min) / x_step)
        yi = int((y[i] - y_min) / y_step)
        u_norm[yi][xi] = np.sqrt(ux_np[i] ** 2 + uy_np[i] ** 2)

    u_norm_pred = np.zeros((len(y_axis), len(x_axis)), float)
    ux_np_pred = ux_pred
    uy_np_pred = uy_pred
    for i in range(len(x)):
        xi = int((x[i] - x_min) / x_step)
        yi = int((y[i] - y_min) / y_step)
        u_norm_pred[yi][xi] = np.sqrt(ux_np_pred[i] ** 2 + uy_np_pred[i] ** 2)

    return p_field, p_field_pred, u_norm, u_norm_pred


x, y, t, p, ux, uy = load_data(0)
# model_params = load_model_params()
# model = load_model(model_params).to(device)
# calculate predictions made by the model
# pred = model.forward(x, y, t, p, ux, uy)
# pred_puv = np.transpose(pred['preds'].cpu().detach().numpy())
# pred_loss = pred['loss']
# pred_losses = pred['losses']
# pred_lambdas = pred['lambdas']

# p_pred = pred_puv[0]
# ux_pred = pred_puv[1]
# uy_pred = pred_puv[2]
# x = x.cpu().detach().numpy()
# y = y.cpu().detach().numpy()
# t = t.cpu().detach().numpy()

p_field, p_field_pred, u_norm, u_norm_pred = transform_data(x, y, t, p, ux, uy, p, ux, uy)  # p_pred, ux_pred, uy_pred)
p_field_history = []
p_field_pred_history = []
u_norm_history = []
u_norm_pred_history = []
loss_history = []
epochs_per_frame = 100

fig, ax = plt.subplots(3, 2, figsize=(10, 10))
u_pred_plot = ax[0][0].imshow(u_norm)  # u_norm_pred
ax[0][0].set_title('predicted velocity at t=0')
p_pred_plot = ax[1][0].imshow(p_field)  # p_field_pred
ax[1][0].set_title('predicted pressure at t=0')
u_actual_plot = ax[0][1].imshow(u_norm)
ax[0][1].set_title('actual velocity at t=0')
p_actual_plot = ax[1][1].imshow(p_field)
ax[1][1].set_title('actual pressure at t=0')
loss_plot, = ax[2][0].plot(loss_history)
ax[2][0].set_yscale('log')
ax[2][0].set_title('loss')
ax[2][0].set_xlabel('epochs')
ax[2][0].set_ylabel('loss')

def update_plot(p_field, p_field_pred, u_norm, u_norm_pred, loss, plot=False):
    p_field_history.append(p_field)
    p_field_pred_history.append(p_field_pred)
    u_norm_history.append(u_norm)
    u_norm_pred_history.append(u_norm_pred)

    u_pred_plot.set_data(u_norm_pred)
    p_pred_plot.set_data(p_field_pred)
    u_actual_plot.set_data(u_norm)
    p_actual_plot.set_data(p_field)

    loss_history.append(loss)

    if plot:
        loss_plot.set_data(epochs_per_frame * np.arange(0, len(loss_history)), loss_history)
        ax[2][0].set_xlim(0, epochs_per_frame * len(loss_history))
        ax[2][0].set_ylim(min(loss_history), max(loss_history))
        fig.suptitle('epochs=' + str(len(p_field_history) * epochs_per_frame))
        fig.canvas.draw()
        plt.pause(0.01)

def init():
    u_pred_plot.set_data(u_norm_pred_history[0])
    p_pred_plot.set_data(p_field_pred_history[0])
    u_actual_plot.set_data(u_norm_history[0])
    p_actual_plot.set_data(p_field_history[0])
    loss_plot.set_data(epochs_per_frame * np.arange(0, len(loss_history)), loss_history)
    return u_pred_plot, u_actual_plot, p_pred_plot, p_actual_plot

def update(frame):
    u_pred_plot.set_data(u_norm_pred_history[frame])
    p_pred_plot.set_data(p_field_pred_history[frame])
    u_actual_plot.set_data(u_norm_history[frame])
    p_actual_plot.set_data(p_field_history[frame])
    loss_plot.set_data(epochs_per_frame * np.arange(0, frame), loss_history[:frame])
    if frame > 0:
        ax[2][0].set_xlim(0, epochs_per_frame * frame)
        ax[2][0].set_ylim(min(loss_history[:frame]), max(loss_history[:frame]))
    fig.suptitle('epochs=' + str(frame * epochs_per_frame))
    return u_pred_plot, u_actual_plot, p_pred_plot, p_actual_plot

def create_animation(save=False, filename='pinn_ns_result.gif'):
    ani = animation.FuncAnimation(fig, update, frames=range(len(u_norm_pred_history)), init_func=init, interval=100, blit=True)
    if save:
        ani.save(filename, writer='pillow', fps=10)
    return ani


if __name__ == '__main__':
    for i in range(0, 200):
        x, y, t, p, ux, uy = load_data(i)
        # pred = model.forward(x, y, t, p, ux, uy)
        # p_pred, ux_pred, uy_pred = np.transpose(pred['preds'].cpu().detach().numpy())
        p_field, p_field_pred, u_norm, u_norm_pred = transform_data(x, y, t, p, ux, uy, p, ux, uy)  # p_pred, ux_pred, uy_pred)
        update_plot(p_field, p_field_pred, u_norm, u_norm_pred)
