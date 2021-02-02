#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 17:48:00 2020

@author: john
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


labels = [f'a = 0.4 $R_H$', 'a = 0.2 $R_H$', 'a = 0.1 $R_H$',]
bound = [76, 85, 92]
swapped = [15, 31, 16]
disrupted = [20, 1, 0]
collisions = [9, 3, 12]

x = np.arange(len(labels))  # the label locations
width = 0.12  # the width of the bars

fig, ax = plt.subplots(1, figsize=(6, 5))
rects1 = sns.barplot(x - width*1.5, bound, label='Bound')
rects2 = sns.barplot(x - width/2, swapped, label='Swapped')
# rects2 = ax.bar(x + width/2, disrupted, width, label='Disrupted')
# rects2 = ax.bar(x + width*1.5, collisions, width, label='Collisions')

ax.set_ylabel('Total')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0,120)
ax.legend()

# fig.savefig('./img/totals_1.pdf', bbox_inches='tight')

# %%
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


labels = [f'a = 0.4 $R_H$', 'a = 0.2 $R_H$', 'a = 0.1 $R_H$',]
bound = [74, 85, 96]
collisions = [7, 5, 7]
disrupted = [33, 18, 5]
swapped = [6, 14, 12]

x = np.arange(len(labels))  # the label locations
width = 0.12  # the width of the bars

fig, ax = plt.subplots(1, figsize=(6, 5))
rects1 = ax.bar(x - width*1.5, bound, width, label='Bound')
rects2 = ax.bar(x - width/2, swapped, width, label='Swapped')
rects2 = ax.bar(x + width/2, disrupted, width, label='Disrupted')
rects2 = ax.bar(x + width*1.5, collisions, width, label='Collisions')

ax.set_ylabel('Total')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0,120)
ax.legend()

fig.savefig('./img/totals_2.pdf', bbox_inches='tight')