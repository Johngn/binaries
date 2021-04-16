import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


labels = ['a = 0.05 $r_\mathrm{H}$', 'a = 0.1 $r_\mathrm{H}$', 'a = 0.2 $r_\mathrm{H}$', 'a = 0.4 $r_\mathrm{H}$']
bound = [0.857, 0.7967, 0.7564, 0.706959]
swapped = [0.0732+0.0659, 0.12087+0.080586, 0.09523+0.11355, 0.08974+0.1043]
# swapped2 = [0.0659, 0.080586, 0.11355, 0.10439]
disrupted = [0.0219, 0.0311, 0.04578, 0.11904]
collisions = [0.047, 0.0677, 0.0366, 0.0457]


# bound = [0.8278, 0.8095, 0.7655, 0.7106]
# swapped = [0.0952+0.0732, 0.0915+0.0622, 0.0805+0.0769, 0.0512+0.0751]
# disrupted = [0.0384, 0.0677, 0.1043, 0.1904]
# collisions = [0.0457, 0.04212, 0.0457, 0.04212]



x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars
# color=['#003f5c','#58508d','#bc5090','#ff6361'], edgecolor="black")
fig, ax = plt.subplots(1, figsize=(5, 4))
rects1 = ax.bar(x - width*1.5, bound, width, label='Bound', color="#444e86", edgecolor="black")
rects2 = ax.bar(x - width/2, swapped, width, label='Swapped', color="#955196", edgecolor="black")
# rects2 = ax.bar(x, swapped2, width, label='Sec-Imp', color="#dd5182", edgecolor="black")
rects3 = ax.bar(x + width/2, disrupted, width, label='Disrupted', color="#ff6e54", edgecolor="black")
rects4 = ax.bar(x + width*1.5, collisions, width, label='Collided', color="#ffa600", edgecolor="black")

ax.grid(color='black', linestyle='--', linewidth=1, axis='y', alpha=0.2)
# ax.set_ylabel('Number of encounter')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0,1.05)
ax.legend()

fig.savefig('./img/totals_equalmass.pdf', bbox_inches='tight')