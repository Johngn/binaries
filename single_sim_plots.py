import numpy as np
import matplotlib.pyplot as plt

n_encounters = 100
x_ticks = np.linspace(0,n_encounters, n_encounters*1000)
fig, axes = plt.subplots(1, figsize=(10,5))

for i in range(5):
    data = np.loadtxt(f"single_sim_{i}")
    e = data[:,0]
    
    e_mavg = [np.mean(e[ii-500:ii]) for ii in range(len(e))]
    
    axes.plot(x_ticks, e, lw=1)
    # axes.plot(x_ticks, e_mavg, lw=1, color="red", label="moving average over two orbits")
    axes.set_ylim(0,1)
    axes.set_xlim(0,n_encounters)
    axes.set_ylabel("eccentricity")
    axes.set_xlabel("cumulative encounters")
    axes.set_title(f'a = 0.2 R$_H$', y=1, pad=15, fontdict={'fontsize': 14})
    axes.grid()
    # axes.legend()

# fig.savefig(f'./img/eccentricity_many_encounters_wide3.pdf', bbox_inches='tight')
# %%

fig, axes = plt.subplots(1, figsize=(10,5))

for i in range(5):
    # i = 3
    data = np.loadtxt(f"single_sim_{i}")
    a = data[:,1]
    
    axes.plot(x_ticks, a, lw=1)
    # axes.set_ylim(0,1)
    axes.set_xlim(0,n_encounters)
    axes.set_ylabel("semi-major axis")
    axes.set_xlabel("cumulative encounters")
    axes.set_title(f'a = 0.2 R$_H$', y=1, pad=15, fontdict={'fontsize': 14})
    axes.grid()

# fig.savefig(f'./img/a_many_encounters_wide3.pdf', bbox_inches='tight')