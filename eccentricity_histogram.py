import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

e = np.array([0.4732, 0.507, 0.021, 0.18, 0.546, 0.03, 0.008, 0.470, 0.026, 0.249, 0.0009, 0.20, 0.010, 0.014, 0.012, 
              0.0062, 0.344, 0.24, 0.588, 0.022, 0.024, 0.418, 0.1494, 0.85, 0.563, 0.02, 0.69, 0.095, 0.019, 0.819, 0.368, 0.30, 0.33,
              0.489, 0.334, 0.464, 0.556, 0.71, 0.663, 0.516, 0.38, 0.438, 0.21, 0.275, 0.896])

bins = 30

fig, ax = plt.subplots(1, figsize=(5,3.5))


# sns.distplot(e, bins=bins, kde=False, norm_hist=False,
                  # hist_kws={"histtype": "step", "linewidth": 1,
                            # "alpha": 1, "color": "black"})
sns.distplot(e, bins=bins, kde=False, color="teal", norm_hist=False,)
# ax.set_title(r'a = 0.4 R${_h}$')
# ax.set_xlim(0,1)
ax.set_xlabel('Eccentricity')

plt.savefig(f"./img/eccentricity_histogram.pdf", bbox_inches='tight')

# %%
a = np.array([
        0.02425, 0.0115, 0.00364, 0.00314, 0.0326, 0.00667, 0.0071, 0.00916, 0.00351, 0.0581, 0.004311 , 0.00329, 0.00233 ,0.00655, 0.00523, 0.004724, 0.0182, 0.035, 0.099, 0.00232, 0.00383, 0.01855, 
        0.0955, 0.090, 0.0167, 0.00318, 0.0178, 0.0253 ,0.0067, 0.04839 ,0.01448, 0.0179, 0.166, 0.01097 ,0.00502, 0.223, 0.01687, 0.0359, 0.0320, 0.0187, 0.145, 0.0148, 0.155, 0.0884, 0.081
    ])

bins = 30

fig, ax = plt.subplots(1, figsize=(5,3.5))


# sns.distplot(a, bins=bins, kde=False, norm_hist=False,
#                   hist_kws={"histtype": "step", "linewidth": 1,
#                             "alpha": 1, "color": "black"})
sns.distplot(a, bins=bins, kde=False, color="teal", norm_hist=False,)

ax.axvline(0.05, color='grey', linewidth=1, ls='--')
# ax.set_title(r'a = 0.4 R${_h}$')
# ax.set_xlim(0,1)
ax.set_xlabel('Mutual orbital separation [$r_\mathrm{H}$]')

plt.savefig(f"./img/mutualorbitseparation_histogram.pdf", bbox_inches='tight')

# %%
fig, ax = plt.subplots(1, figsize=(5,4))
ax.set_xlabel('Collision speed [m/s]')
ax.set_ylabel(r'Collision angle [$^\circ$]')
plt.hist2d(a, e, bins=10, cmap='plasma', density=True)
# plt.savefig(f"./img/collision_hist2d_{sim_name}.pdf", bbox_inches="tight")