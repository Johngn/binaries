import numpy as np
import matplotlib.pyplot as plt

a_bins =    [0.4,   0.35,   0.3,    0.25,   0.2,    0.15,   0.1, 0.05]
sim_times = [0.04,  0.04,   0.0425, 0.045, 0.05,   0.1,    30,  60*30]

fig, axes = plt.subplots(1, figsize=(4, 3))
axes.set_xlabel("$a/r_\mathrm{H}$")
axes.set_ylabel("Simulation time [$\mathrm{s}$]")
# axes.set_ylim(-lim,lim)
# axes.set_xlim(-lim,lim)
# axes.set_xscale('log')
axes.set_yscale('log')
axes.plot(a_bins, sim_times, marker='D', color='slateblue', lw=1, ms=4)

# axes.grid()
# axes.legend()
fig.savefig(f'./img/sim_time.pdf', bbox_inches='tight')