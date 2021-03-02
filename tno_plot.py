import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

classical = np.logical_and(data_clean[:,2] > 41, data_clean[:,2] < 48.7) # select objects with range of 'a'
low_ecc = data_clean[:,1] < 0.3
low_inc = data_clean[:,0] < 37

low_ecc_inc = np.logical_and(low_ecc, low_inc)
cold = np.logical_and(data_clean[:,0] < 5, low_ecc_inc) # low inclination
hot = np.logical_and(data_clean[:,0] > 5, low_ecc_inc) # high inclination

cold_classical = np.logical_and(cold, classical)
data_cold_classical = data_clean[cold_classical]

hot_classical = np.logical_and(hot, classical)
data_hot_classical = data_clean[hot_classical]

df = pd.DataFrame(data_clean, columns=('i','e','a'))
df.to_csv('./data/tnos_clean', index=None)
# %%
data_clean = pd.read_csv('./data/tnos_clean')

fig, ax = plt.subplots(2, figsize=(8,8))
fig.subplots_adjust(hspace=0)

s = 25
alpha = 1

ax[0].scatter(data_clean['a'],data_clean['e'], s=s, alpha=alpha, color="dimgray", edgecolors="black")
ax[0].scatter(data_hot_classical[:,2],data_hot_classical[:,1], alpha=alpha, s=s, color='darkturquoise', label='Hot classicals', edgecolors='black')
ax[0].scatter(data_cold_classical[:,2],data_cold_classical[:,1], alpha=alpha, s=s, color='red', label='Cold classicals', edgecolors='black')
ax[0].scatter(resonant_objs[:,2],resonant_objs[:,1], alpha=alpha, s=s, color='limegreen', label='Plutinos', edgecolors='black')
# ax[0].scatter(39.482, 0.2488, s=200, label='Pluto', color='darkorange', edgecolors='black')
# ax[0].scatter(30.11, 0.008678, s=150,  label='Neptune', color='darkviolet', edgecolors='black')
ax[0].set_ylim(0, 0.4)
ax[0].set_xlim(35, 60)
ax[0].set_xticks(np.arange(0))
# ax[0].axvline(44)
ax[0].set_ylabel(r'Eccentricity')
# ax[0].set_xlabel('Semi-major axis [AU]')

ax[1].scatter(data_clean['a'],data_clean['i'], s=s, alpha=alpha, color='dimgray', edgecolors='black')
ax[1].scatter(data_cold_classical[:,2],data_cold_classical[:,0], s=s, alpha=alpha, color='red', edgecolors='black')
ax[1].scatter(data_hot_classical[:,2],data_hot_classical[:,0], s=s, alpha=alpha, color='darkturquoise', edgecolors='black')
ax[1].scatter(resonant_objs[:,2],resonant_objs[:,0], s=s, alpha=alpha, color='limegreen', edgecolors='black')
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