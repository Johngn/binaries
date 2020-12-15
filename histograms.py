#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 17:48:00 2020

@author: john
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


labels = [f'a = 0.4 $R_H$', 'a = 0.2 $R_H$', 'a = 0.1 $R_H$',]
bound = [76, 34, 92]
swapped = [15, 32, 16]
disrupted = [20, 32, 0]
collisions = [9, 32, 12]

x = np.arange(len(labels))  # the label locations
width = 0.1  # the width of the bars

fig, ax = plt.subplots(1, figsize=(5, 5))
rects1 = ax.bar(x - width*1.5, bound, width, label='Bound')
rects2 = ax.bar(x - width/2, swapped, width, label='Swapped')
rects2 = ax.bar(x + width/2, disrupted, width, label='Disrupted')
rects2 = ax.bar(x + width*1.5, collisions, width, label='Collisions')

ax.set_ylabel('Total')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.savefig('./img/totals_histogram.pdf', bbox_inches='tight')