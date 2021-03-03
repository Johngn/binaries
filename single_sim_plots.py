import numpy as np
import matplotlib.pyplot as plt

n_encounters = 100
x_ticks = np.linspace(0,n_encounters, n_encounters*1000)
fig, axes = plt.subplots(2,1, figsize=(10,10))
fig.subplots_adjust(hspace=0.15)

for i in range(10):
    data = np.loadtxt(f"./data/single_sim_ecc2_{i}")
    a = data[:,0]
    e = data[:,1]
    
    e_mavg = [np.mean(e[ii-500:ii]) for ii in range(len(e))]
    
    # axes.plot(x_ticks, e, lw=1)
    axes[0].plot(x_ticks, a, lw=0.25)
    axes[0].set_ylim(0,0.3)
    axes[0].set_xlim(0,n_encounters)
    axes[0].set_ylabel("Semi-major axis [AU]")
    # axes[0].set_xlabel("Cumulative encounters")
    # axes[0].set_title(f'a = 0.2 R$_H$', y=1, pad=15, fontdict={'fontsize': 14})
    # axes[0].grid()
    # axes[0].set_xticks(np.arange(0))
    # axes.legend()


    
    axes[1].plot(x_ticks, e_mavg, lw=2)
    axes[1].set_ylim(0.15, 0.22)
    axes[1].set_xlim(0,n_encounters)
    axes[1].set_ylabel("Eccentricity")
    axes[1].set_xlabel("Cumulative encounters")
    # axes[1].set_yticks(np.arange(0,41,0.05))
    # axes[1].set_title(f'a = 0.2 R$_H$', y=1, pad=15, fontdict={'fontsize': 14})
    # axes[1].grid()

fig.savefig(f'./img/many_encounters_wide_10_ecc2.png', bbox_inches='tight')