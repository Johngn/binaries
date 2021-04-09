import numpy as np
import matplotlib.pyplot as plt

n_encounters = 100000
noutputs = 10

x_ticks = np.linspace(0,n_encounters, n_encounters*noutputs)

sim_name = "single_wide_equalmass_5inc"

n_sims = 5

a_ended = np.ones((n_sims,2))*np.nan
e_ended = np.ones((n_sims,2))*np.nan
i_ended = np.ones((n_sims,2))*np.nan

a_mavg = np.zeros((n_encounters*noutputs, n_sims))
e_mavg = np.zeros((n_encounters*noutputs, n_sims))
i_mavg = np.zeros((n_encounters*noutputs, n_sims))

for i in range(n_sims):
    data = np.loadtxt(f"./data/{sim_name}_{i}")
    a = data[:,0]
    a[a <= 0] = np.nan
    a[a >= 1] = np.nan
    if (np.isnan(a).any()):
        a_ended[i] = [np.where(np.isnan(a))[0][0]-1, a[np.where(np.isnan(a))[0][0]-1]]
    
    e = data[:,1]
    e[e < 0] = np.nan
    e[e > 1] = np.nan
    if (np.isnan(e).any()):
        e_ended[i] = [np.where(np.isnan(e))[0][0]-1, e[np.where(np.isnan(e))[0][0]-1]]
    
    inc = np.rad2deg(data[:,2])
    inc[inc < 0] = np.nan
    inc[inc >= 20] = np.nan
    if (np.isnan(inc).any()):
        i_ended[i] = [np.where(np.isnan(inc))[0][0]-1, inc[np.where(np.isnan(inc))[0][0]-1]]
    
    a_mavg[:,i] = [np.mean(a[ii-100:ii]) for ii in range(len(a))]
    e_mavg[:,i] = [np.mean(e[ii-100:ii]) for ii in range(len(e))]
    i_mavg[:,i] = [np.mean(inc[ii-100:ii]) for ii in range(len(inc))]
    
    # a_mavg[:,i] = a
    # e_mavg[:,i] = e
    # i_mavg[:,i] = inc
    
# %%
fig, axes = plt.subplots(3,1, figsize=(10,7))
# fig.subplots_adjust(hspace=0.15)

for i in range(n_sims):
    
    axes[0].plot(x_ticks, a_mavg[:,i], lw=1.5, alpha=1)
    # axes[0].scatter(a_ended[i][0]/10, a_ended[i][1], s=100, marker='X')
    # axes[0].scatter(60285.0,0.6666479, s=100)
    axes[0].set_ylim(0, 1)
    axes[0].set_xlim(0,n_encounters)
    axes[0].set_ylabel(f"Mutual orbital separation [r$_H$]")
    # axes[0].grid()
    # axes[0].set_xticks(np.arange(0))
    axes[0].set_xticklabels('')
    # axes.legend()
    
    axes[1].plot(x_ticks, e_mavg[:,i], lw=1.5, alpha=1)
    # axes[1].scatter(e_ended[i][0]/10, e_ended[i][1], s=100, marker='X')
    axes[1].set_ylim(0, 1)
    axes[1].set_xlim(0,n_encounters)
    axes[1].set_ylabel("Eccentricity")
    # axes[1].set_xlabel("Cumulative encounters")
    axes[1].set_xticklabels('')
    # axes[1].set_yticks([0.1,0.2,0.3,0.4])
    # axes[1].set_yticks(np.arange(0,41,0.05))
    # axes[1].set_title(f'a = 0.2 R$_H$', y=1, pad=15, fontdict={'fontsize': 14})
    # axes[1].grid()
    
    
    
    axes[2].plot(x_ticks, i_mavg[:,i], lw=1.5, alpha=1)
    # axes[2].scatter(i_ended[i][0]/10, i_ended[i][1], s=100, marker='X')
    axes[2].set_ylim(0, 10)
    axes[2].set_xlim(0,n_encounters)
    axes[2].set_ylabel("Inclination [$^{\circ}$]")
    axes[2].set_xlabel("Cumulative encounters")
    # axes[2].grid()

fig.savefig(f'./img/single_follow_{sim_name}.pdf', bbox_inches='tight')
# %%
r_a = 0.1
r_p = 0.01

e = (r_a - r_p)/(r_a + r_p)
