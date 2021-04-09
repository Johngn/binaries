import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

tnos = pd.read_table('./data/TNOs.txt', names='-', skiprows=1)
tnos = tnos['-'] # each row is just one long string
tnos1 = [item.split(' ') for i, item in enumerate(tnos)] # split into many strings
str_list = [list(filter(None, item)) for i, item in enumerate(tnos1)] # remove empty strings

data = np.array([item[9:13] for i, item in enumerate(str_list)])

# split into half that have different formats
data_high = data[:2070,0:3]
data_low = data[2070:,1:4]

data_high = np.delete(data_high, (1412,), axis=0).astype(float) # remove row with bad data
data_low = np.delete(data_low, (7,), axis=0).astype(float) # remove row with bad data

data_clean = np.vstack((data_high, data_low))

resonant = np.logical_and(data_clean[:,2] > 39, data_clean[:,2] < 40.3)
resonant_objs = data_clean[resonant]

classical = np.logical_and(data_clean[:,2] > 41, data_clean[:,2] < 48) # select objects with range of 'a'
low_ecc = data_clean[:,1] < 0.3
low_inc = data_clean[:,0] < 37

low_ecc_inc = np.logical_and(low_ecc, low_inc)
cold = np.logical_and(data_clean[:,0] < 5, low_ecc_inc) # low inclination
hot = np.logical_and(data_clean[:,0] > 5, low_ecc_inc) # high inclination
cold = data_clean[:,0] < 5 # low inclination
hot = data_clean[:,0] > 5 # high inclination

data_classical = data_clean[classical]

cold_classical = np.logical_and(cold, classical)
data_cold_classical = data_clean[cold_classical]

hot_classical = np.logical_and(hot, classical)
data_hot_classical = data_clean[hot_classical]

df = pd.DataFrame(data_clean, columns=('i','e','a'))
df.to_csv('./data/tnos_clean', index=None)
# %%
data_clean = pd.read_csv('./data/tnos_clean')

fig, ax = plt.subplots(2, figsize=(7,7))
fig.subplots_adjust(hspace=0)

s = 6
alpha = 1
lw = 0.1
edgecolor = ''

ax[0].scatter(data_clean['a'],data_clean['e'], s=s, alpha=alpha, color="dimgray", edgecolors=edgecolor, linewidth=lw)
ax[0].scatter(data_hot_classical[:,2],data_hot_classical[:,1], alpha=alpha, s=s, color='blue', label='Hot classicals', edgecolors=edgecolor, linewidth=lw)
ax[0].scatter(data_cold_classical[:,2],data_cold_classical[:,1], alpha=alpha, s=s, color='red', label='Cold classicals', edgecolors=edgecolor, linewidth=lw)
ax[0].scatter(resonant_objs[:,2],resonant_objs[:,1], alpha=alpha, s=s, color='limegreen', label='Plutinos', edgecolors=edgecolor, linewidth=lw)
# ax[0].scatter(39.482, 0.2488, s=200, label='Pluto', color='darkorange', edgecolors='black')
# ax[0].scatter(30.11, 0.008678, s=150,  label='Neptune', color='darkviolet', edgecolors='black')
ax[0].set_ylim(0, 0.4)
ax[0].set_xlim(35, 60)
ax[0].set_xticks(np.arange(0))
# ax[0].axvline(44)
ax[0].set_ylabel(r'Eccentricity')
# ax[0].set_xlabel('Semi-major axis [AU]')

ax[1].scatter(data_clean['a'],data_clean['i'], s=s, alpha=alpha, color='dimgray', edgecolors=edgecolor, linewidth=lw)
ax[1].scatter(data_cold_classical[:,2],data_cold_classical[:,0], s=s, alpha=alpha, color='red', edgecolors=edgecolor, linewidth=lw)
ax[1].scatter(data_hot_classical[:,2],data_hot_classical[:,0], s=s, alpha=alpha, color='blue', edgecolors=edgecolor, linewidth=lw)
ax[1].scatter(resonant_objs[:,2],resonant_objs[:,0], s=s, alpha=alpha, color='limegreen', edgecolors=edgecolor, linewidth=lw)
# ax[1].scatter(39.482, 17.16, s=200, label='Pluto', color='darkorange', edgecolors='black')
# ax[1].scatter(30.11, 1.77, s=150,  label='Neptune', color ='darkviolet', edgecolors='black')
ax[1].set_ylim(0, 45)
ax[1].set_xlim(35,60)
ax[1].set_yticks(np.arange(0,41,5))
# ax[1].axvline(44)
ax[1].set_ylabel(r'Inclination [$^\circ$]')
ax[1].set_xlabel('Semi-major axis [AU]')
# ax.grid()
ax[0].legend()

fig.savefig('./img/cckbos.pdf', bbox_inches='tight')

# %%
data_cold_classical_no_zero_ecc = data_cold_classical[data_cold_classical[:,1] > 0]
data_hot_classical_no_zero_ecc = data_hot_classical[data_hot_classical[:,1] > 0]

bins = 10

fig, ax = plt.subplots(1, figsize=(5,4))

sns.distplot(data_cold_classical_no_zero_ecc[:,0], bins=bins, kde=False, norm_hist=True,
                   hist_kws={"histtype": "step", "linewidth": 4, "color":"red"}, label="Cold-classicals")
# sns.distplot(data_cold_classical_no_zero_ecc[:,0], bins=bins, kde=False, color="red", norm_hist=True, label="Cold-classicals")

sns.distplot(data_hot_classical_no_zero_ecc[:,0], bins=20, kde=False, norm_hist=True,
                   hist_kws={"histtype": "step", "linewidth": 4, "color": "blue"}, label="Hot-classicals")
# sns.distplot(data_hot_classical_no_zero_ecc[:,0], bins=bins, kde=False, color="darkturquoise", norm_hist=True, label="Hot-classicals")


sns.distplot(np.random.rayleigh(2, 1000000), bins=200, kde=True, hist=False,
              kde_kws={"color": "darkred", "lw": 1}, label='$\sigma$ = 2')
sns.distplot(np.random.rayleigh(10, 100000), bins=200, kde=True, hist=False,
             kde_kws={"color": "blue", "lw": 1}, label='$\sigma$ = 10')

ax.set_xlim(-2, 50)
plt.legend()
ax.set_xlabel('Inclination [$^{\circ}$]')
plt.savefig(f"./img/inclination_distplot.pdf", bbox_inches='tight')

# %%

bins = 30

fig, ax = plt.subplots(1, figsize=(5,4))

sns.distplot(np.random.rayleigh(0.05, 1000000), bins=200, kde=True, hist=False,
             kde_kws={"color": "darkred", "lw": 1}, label='$\sigma$ = 0.05')
sns.distplot(np.random.rayleigh(0.11, 1000000), bins=200, kde=True, hist=False,
              kde_kws={"color": "blue", "lw": 1}, label='$\sigma$ = 0.1')

sns.distplot(data_cold_classical_no_zero_ecc[:,1], bins=bins, kde=False, norm_hist=True,
                   hist_kws={"histtype": "step", "linewidth": 4, "color":"red"}, label="Cold-classicals")
# sns.distplot(data_cold_classical_no_zero_ecc[:,1], bins=bins, kde=False, color="red", norm_hist=True, label="Cold-classicals")


sns.distplot(data_hot_classical_no_zero_ecc[:,1], bins=bins, kde=False, norm_hist=True,
                   hist_kws={"histtype": "step", "linewidth": 4, "color": "blue"}, label="Hot-classicals")
# sns.distplot(data_hot_classical_no_zero_ecc[:,1], bins=bins, kde=False, color="darkturquoise", norm_hist=True, label="Hot-classicals")
# ax.set_title(r'a = 0.4 R${_h}$')
ax.set_xlim(-0.02,0.4)
plt.legend()
ax.set_xlabel('Eccentricity')
plt.savefig(f"./img/eccentricity_distplot.pdf", bbox_inches='tight')








































