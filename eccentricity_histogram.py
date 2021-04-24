import numpy as np
import matplotlib.pyplot as plt

e = np.array([0.4732, 0.507, 0.021, 0.18, 0.546, 0.03, 0.008, 0.470, 0.026, 0.249, 0.0009, 0.20, 0.010, 0.014, 0.012, 
              0.0062, 0.344, 0.24, 0.588, 0.022, 0.024, 0.418, 0.1494, 0.85, 0.563, 0.02, 0.69, 0.095, 0.019, 0.819, 0.368, 0.30, 0.33,
              0.489, 0.334, 0.464, 0.556, 0.71, 0.663, 0.516, 0.38, 0.438, 0.21, 0.275, 0.896])

bins = 20

fig, ax = plt.subplots(1, figsize=(5,3.5))


sns.distplot(e, bins=bins, kde=False, norm_hist=False,
                  hist_kws={"histtype": "step", "linewidth": 1,
                            "alpha": 1, "color": "black"})
sns.distplot(e, bins=bins, kde=False, color="red", norm_hist=False,)
# ax.set_title(r'a = 0.4 R${_h}$')
# ax.set_xlim(0,1)
ax.set_xlabel('Eccentricity')

plt.savefig(f"./img/eccentricity_histogram.pdf", bbox_inches='tight')