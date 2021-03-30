import numpy as np
import matplotlib.pyplot as plt

# n_encounters = 1000
# noutputs = 100 

x_ticks = np.linspace(0,n_encounters, n_encounters*noutputs)
fig, axes = plt.subplots(2,1, figsize=(10,6))
fig.subplots_adjust(hspace=0.15)
sim_name = "single_sim_highecc"

for i in range(5):
    data = np.loadtxt(f"./data/{sim_name}_{i}")
    a = data[:,0]
    e = data[:,1]
    
    e_mavg = [np.mean(e[ii-1000:ii]) for ii in range(len(e))]
    a_mavg = [np.mean(a[ii-1000:ii]) for ii in range(len(a))]
    
    axes[0].plot(x_ticks, a_mavg, lw=2)
    # axes[0].set_ylim(0)
    axes[0].set_xlim(0,n_encounters)
    axes[0].set_ylabel(f"Semi-major axis [R$_H$]")
    # axes[0].grid()
    # axes[0].set_xticks(np.arange(0))
    # axes.legend()


    
    axes[1].plot(x_ticks, e_mavg, lw=2)
    axes[1].set_ylim(0,0.1)
    axes[1].set_xlim(0,n_encounters)
    axes[1].set_ylabel("Eccentricity")
    axes[1].set_xlabel("Cumulative encounters")
    # axes[1].set_yticks(np.arange(0,41,0.05))
    # axes[1].set_title(f'a = 0.2 R$_H$', y=1, pad=15, fontdict={'fontsize': 14})
    # axes[1].grid()

fig.savefig(f'./img/{sim_name}.pdf', bbox_inches='tight')
