import my_utils_1
import pickle
import numpy as np
import matplotlib.pyplot as plt

x, y, t, p, ux, uy = my_utils_1.load_data(0)
p_field, p_field_pred, u_norm, u_norm_pred = my_utils_1.transform_data(x, y, t, p, ux, uy, p, ux, uy)

ns_samples = [100, 200, 500, 1000, 2000, 5000]
files = [f'./pinn_ns_results/n_sample_{n}/pinn_ns_result_20000.pickle' for n in ns_samples]
preds = [pickle.load(open(file, 'rb')) for file in files]
u_norm_preds = [pred['u_norm_pred'][len(pred['u_norm_pred']) - 1] for pred in preds]
p_field_preds = [pred['p_pred'][len(pred['p_pred']) - 1] for pred in preds]
normalization_constants = [np.mean(p_field_preds[i]) - np.mean(p_field) for i in range(len(ns_samples))]
print(normalization_constants)
p_field_preds = [p_field_preds[i] - normalization_constants[i] for i in range(len(ns_samples))]
u_norm_losses = []
p_losses = []

for i in range(len(ns_samples)):
    u_norm_losses.append(np.mean(np.abs(u_norm_preds[i] - u_norm) / np.abs(u_norm)))
    p_losses.append(np.mean(np.abs(p_field_preds[i] - p_field) / np.abs(p_field)))
plt.plot(u_norm_losses)
plt.xticks(np.arange(0, len(ns_samples)), [str(n) for n in ns_samples])
# plt.yscale('log')
plt.xlabel('Number of samples')
plt.ylabel('Numerical error of velocity norm')
plt.show()
plt.plot(p_losses)
plt.xticks(np.arange(0, len(ns_samples)), [str(n) for n in ns_samples])
# plt.yscale('log')
plt.xlabel('Number of samples')
plt.ylabel('Numerical error of pressure')
plt.show()
