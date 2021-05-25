import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

tnos = pd.read_table('./data/minorplanets.txt', names='-', skiprows=0)
tnos = tnos['-'] # each row is just one long string
tnos1 = [item.split(' ') for i, item in enumerate(tnos)] # split into many strings
str_list = [list(filter(None, item)) for i, item in enumerate(tnos1)] # remove empty strings

data = np.array([item[7:11] for i, item in enumerate(str_list)])
data = np.delete(data, (1631), axis=0)
data = data.astype(float)

a = data[:,3]
e = data[:,1]
i = data[:,0]

classical = np.logical_and(a > 42, a < 48) # select objects with range of 'a'
cold = np.logical_and(classical, np.logical_and(i < 5, e < 0.3))
hot = np.logical_and(classical, np.logical_and(i > 5, e < 0.3))
hot2 = np.logical_and(np.logical_and(a > 41, a < 47), i > 5)

plutinos = np.logical_and(a > 39, a < 40.3)

scattered1 = np.logical_and(np.logical_and(a > 49, a < 54.8), np.logical_and(e > 0.2, e < 0.45))
scattered2 = np.logical_and(np.logical_and(a > 55.7, a < 62), np.logical_and(e > 0.3, e < 0.6))
scattered3 = np.logical_and(np.logical_and(a > 62, a < 72), np.logical_and(e > 0.4, e < 0.6))
scattered4 = np.logical_and(np.logical_and(a > 72, a < 80), np.logical_and(e > 0.45, e < 0.6))

scattered5 = np.logical_and(np.logical_and(a > 34, a < 36), np.logical_and(e > 0.1, e < 0.2))
scattered6 = np.logical_and(np.logical_and(a > 37, a < 49), np.logical_and(e > 0.1, e < 0.2))


aas = np.arange(0.1,100,1)
ees30 = 1-30/aas
ees40 = 1-40/aas

fig, ax = plt.subplots(2, figsize=(7,7))
fig.subplots_adjust(hspace=0)

s = 4
alpha = 1
lw = 0.1
edgecolor = ''

ax[0].plot(ees30, lw=0.5, color='black')
ax[0].plot(ees40, lw=0.5, color='black')

ax[0].scatter(0,0, alpha=alpha, s=20, color='blue', label='Hot classicals')
ax[0].scatter(0,0, alpha=alpha, s=20, color='red', label='Cold classicals')
ax[0].scatter(0,0, alpha=alpha, s=20, color='limegreen', label='Plutinos')
ax[0].scatter(0,0, alpha=alpha, s=20, color='orange', label='Scattered disc')

ax[0].scatter(a,e, s=s, alpha=alpha, color="gray", edgecolors=edgecolor, linewidth=lw)
ax[0].scatter(a[scattered1],e[scattered1], alpha=alpha, s=s, color='orange', edgecolors=edgecolor, linewidth=lw)
ax[0].scatter(a[scattered2],e[scattered2], alpha=alpha, s=s, color='orange', edgecolors=edgecolor, linewidth=lw)
ax[0].scatter(a[scattered3],e[scattered3], alpha=alpha, s=s, color='orange', edgecolors=edgecolor, linewidth=lw)
ax[0].scatter(a[scattered4],e[scattered4], alpha=alpha, s=s, color='orange', edgecolors=edgecolor, linewidth=lw)
ax[0].scatter(a[scattered5],e[scattered5], alpha=alpha, s=s, color='orange', edgecolors=edgecolor, linewidth=lw)
ax[0].scatter(a[scattered6],e[scattered6], alpha=alpha, s=s, color='orange', edgecolors=edgecolor, linewidth=lw)
ax[0].scatter(a[hot],e[hot], alpha=alpha, s=s, color='blue', edgecolors=edgecolor, linewidth=lw)
ax[0].scatter(a[hot2],e[hot2], alpha=alpha, s=s, color='blue', edgecolors=edgecolor, linewidth=lw)
ax[0].scatter(a[cold],e[cold], alpha=alpha, s=s, color='red', edgecolors=edgecolor, linewidth=lw)
ax[0].scatter(a[plutinos],e[plutinos], alpha=alpha, s=s, color='limegreen', edgecolors=edgecolor, linewidth=lw)
# ax[0].scatter(39.482, 0.2488, s=100, label='Pluto', color='darkgreen', edgecolors='')
ax[0].scatter(30.11, 0.008678, s=100,  label='Neptune', color='steelblue', edgecolors='')
ax[0].set_ylim(0, .5)
ax[0].set_xlim(28, 80)
ax[0].set_xticks(np.arange(0))
# ax[0].axvline(44)
ax[0].set_ylabel(r'Eccentricity')
# ax[0].set_xlabel('Semi-major axis [AU]')
ax[0].legend()


ax[1].scatter(a,i, s=s, alpha=alpha, color='dimgray', edgecolors=edgecolor, linewidth=lw)
ax[1].scatter(data[scattered1,3],data[scattered1,0], alpha=alpha, s=s, color='orange', edgecolors=edgecolor, linewidth=lw)
ax[1].scatter(data[scattered2,3],data[scattered2,0], alpha=alpha, s=s, color='orange', edgecolors=edgecolor, linewidth=lw)
ax[1].scatter(data[scattered3,3],data[scattered3,0], alpha=alpha, s=s, color='orange', edgecolors=edgecolor, linewidth=lw)
ax[1].scatter(data[scattered4,3],data[scattered4,0], alpha=alpha, s=s, color='orange', edgecolors=edgecolor, linewidth=lw)
ax[1].scatter(data[scattered5,3],data[scattered5,0], alpha=alpha, s=s, color='orange', edgecolors=edgecolor, linewidth=lw)
ax[1].scatter(data[scattered6,3],data[scattered6,0], alpha=alpha, s=s, color='orange', edgecolors=edgecolor, linewidth=lw)
ax[1].scatter(data[cold,3],data[cold,0], s=s, alpha=alpha, color='red', edgecolors=edgecolor, linewidth=lw)
ax[1].scatter(data[hot,3],data[hot,0], s=s, alpha=alpha, color='blue', edgecolors=edgecolor, linewidth=lw)
ax[1].scatter(data[hot2,3],data[hot2,0], s=s, alpha=alpha, color='blue', edgecolors=edgecolor, linewidth=lw)
ax[1].scatter(data[plutinos,3],data[plutinos,0], s=s, alpha=alpha, color='limegreen', edgecolors=edgecolor, linewidth=lw)
# ax[1].scatter(39.482, 17.16, s=100, label='Pluto', color='darkgreen', edgecolors='')
ax[1].scatter(30.11, 1.77, s=100,  label='Neptune', color ='steelblue', edgecolors='')
ax[1].set_ylim(0, 50)
ax[1].set_xlim(28,80)
ax[1].set_yticks(np.arange(0,46,5))
# ax[1].axvline(44)
ax[1].set_ylabel(r'Inclination [$^\circ$]')
ax[1].set_xlabel('Semi-major axis [AU]')
# ax.grid()

ax[0].axvline(36.4, color='black', linewidth=0.7, label="3:2", ls='--')
ax[0].axvline(39.4, color='black', linewidth=0.7, label="3:2", ls='--')
ax[0].axvline(42.2, color='black', linewidth=0.7, label="5:3", ls='--')
ax[0].axvline(43.7, color='black', linewidth=0.7, label="7:4", ls='--')
ax[0].axvline(47.8, color='black', linewidth=0.7, label="2:1", ls='--')
ax[0].axvline(55.3, color='black', linewidth=0.7, label="5:2", ls='--')

ax[1].axvline(36.4, color='black', linewidth=0.7, label="3:2", ls='--')
ax[1].axvline(39.4, color='black', linewidth=0.7, label="3:2", ls='--')
ax[1].axvline(42.2, color='black', linewidth=0.7, label="5:3", ls='--')
ax[1].axvline(43.7, color='black', linewidth=0.7, label="7:4", ls='--')
ax[1].axvline(47.8, color='black', linewidth=0.7, label="2:1", ls='--')
ax[1].axvline(55.3, color='black', linewidth=0.7, label="5:2", ls='--')

# fig.savefig('./img/cckbos.pdf', bbox_inches='tight')

# %%
hot = np.logical_and(classical, i > 5)
bins = 8

fig, ax = plt.subplots(1, figsize=(5,4))

sns.distplot(i[cold], bins=bins, kde=False, norm_hist=True,
                    hist_kws={"histtype": "step", "linewidth": 3, "color":"red"}, label="Cold classicals")

sns.distplot(i[hot], bins=50, kde=False, norm_hist=True,
                    hist_kws={"histtype": "step", "linewidth": 3, "color": "blue"}, label="Hot classicals")


sns.distplot(np.random.rayleigh(2, 1000000), bins=200, kde=True, norm_hist=False, hist=False,
                kde_kws={"color": "darkred", "lw": 1},
                hist_kws={"histtype": "step", "linewidth": 4, "color":"red"})
sns.distplot(np.random.rayleigh(10, 1000000), bins=200, kde=True, norm_hist=True, hist=False,
                kde_kws={"color": "blue", "lw": 1},
                hist_kws={"histtype": "step", "linewidth": 4, "color":"red"})

ax.set_xlim(0, 50)
plt.legend()
plt.grid()
ax.set_xlabel('Inclination [$^{\circ}$]')
ax.set_ylabel('N')
plt.savefig(f"./img/inclination_distplot.pdf", bbox_inches='tight')

# %%

bins = 20

fig, ax = plt.subplots(1, figsize=(5,4))

sns.distplot(e[cold], bins=bins, kde=False, norm_hist=True,
                    hist_kws={"histtype": "step", "linewidth": 3, "color":"red"}, label="Cold classicals")

sns.distplot(e[hot], bins=60, kde=False, norm_hist=True,
                    hist_kws={"histtype": "step", "linewidth": 3, "color": "blue"}, label="Hot classicals")

sns.distplot(np.random.rayleigh(0.05, 1000000), bins=200, kde=True, norm_hist=False, hist=False,
                kde_kws={"color": "darkred", "lw": 1},
                hist_kws={"histtype": "step", "linewidth": 4, "color":"red"})
sns.distplot(np.random.rayleigh(0.11, 1000000), bins=200, kde=True, norm_hist=True, hist=False,
                kde_kws={"color": "blue", "lw": 1},
                hist_kws={"histtype": "step", "linewidth": 4, "color":"red"})
# ax.set_title(r'a = 0.4 R${_h}$')
ax.set_xlim(-0.02,0.35)
plt.legend()
plt.grid()
ax.set_xlabel('Eccentricity')
ax.set_ylabel('N')
plt.savefig(f"./img/eccentricity_distplot.pdf", bbox_inches='tight')

# %%
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

fig, axe = plt.subplots()

x0 = data_cold_classical_no_zero_ecc[:,0]
loc, scale = stats.rayleigh.fit(x0)
xl = np.linspace(x0.min(), x0.max(), 100)
axe.plot(xl, stats.rayleigh(scale=scale, loc=loc).pdf(xl), c='red')
sns.distplot(data_cold_classical_no_zero_ecc[:,0], bins=15, kde=False, norm_hist=True,
                   hist_kws={"histtype": "step", "linewidth": 3, "color":"red"}, label="Cold-classicals")


x0 = data_hot_classical_no_zero_ecc[:,0]
loc, scale = stats.rayleigh.fit(x0)
xl = np.linspace(x0.min(), x0.max(), 100)
axe.plot(xl, stats.rayleigh(scale=scale, loc=loc).pdf(xl), c='blue')
sns.distplot(data_hot_classical_no_zero_ecc[:,0], bins=30, kde=False, norm_hist=True,
                   hist_kws={"histtype": "step", "linewidth": 3, "color":"blue"}, label="Hot-classicals")

axe.set_xlabel('Inclination [$^{\circ}$]')
# axe.set_xlim(-0.02,0.4)
axe.legend()
axe.grid()

# plt.savefig(f"./img/inclination_distplot.pdf", bbox_inches='tight')

# %%
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

fig, axe = plt.subplots()

x0 = e[cold]
loc, scale = stats.rayleigh.fit(x0)
xl = np.linspace(x0.min(), x0.max(), 100)
axe.plot(xl, stats.rayleigh(scale=scale, loc=loc).pdf(xl), c='red')
sns.distplot(e[cold], bins=25, kde=False, norm_hist=True,
                   hist_kws={"histtype": "step", "linewidth": 3, "color":"red"}, label="Cold-classicals")


x0 = e[hot]
loc, scale = stats.rayleigh.fit(x0)
xl = np.linspace(x0.min(), x0.max(), 100)
axe.plot(xl, stats.rayleigh(scale=scale, loc=loc).pdf(xl), c='blue')
sns.distplot(e[hot], bins=35, kde=False, norm_hist=True,
                   hist_kws={"histtype": "step", "linewidth": 3, "color":"blue"}, label="Hot-classicals")

axe.set_xlabel('Eccentricity')
axe.set_xlim(-0.02,0.4)
axe.legend()
# axe.grid()

# plt.savefig(f"./img/eccentricity_distplot.pdf", bbox_inches='tight')
































