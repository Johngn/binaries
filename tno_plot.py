#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 18:48:29 2020

@author: john
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

tnos = pd.read_table('./TNOs.txt', names='-', skiprows=1)
tnos = tnos['-']
tnos1 = [item.split(' ') for i, item in enumerate(tnos)]
str_list = [list(filter(None, item)) for i, item in enumerate(tnos1)]

data = np.array([item[9:13] for i, item in enumerate(str_list)])

data_high = data[:2070,0:3]
data_low = data[2070:,1:4]

data_high_float_filtered = np.delete(data_high, (1413,), axis=0).astype(float)
data_low_float_filtered = np.delete(data_low, (8,), axis=0).astype(float)

data_clean = np.vstack((data_high_float_filtered, data_low_float_filtered))

classical = np.logical_and(data_clean[:,2] > 41, data_clean[:,2] < 48.7)
cold = data_clean[:,0] < 5
hot = data_clean[:,0] > 5

cold_classical = np.logical_and(cold, classical)
data_cold_classical = data_clean[cold_classical]

hot_classical = np.logical_and(hot, classical)
data_hot_classical = data_clean[hot_classical]



df = pd.DataFrame(data_clean, columns=('i','e','a'))
df.to_csv('./tnos_clean', index=None)

fig, ax = plt.subplots(1, figsize=(10,10))

ax.scatter(data_clean[:,2],data_clean[:,0], s=15, alpha=0.7)
# ax.scatter(data_cold_classical[:,2],data_cold_classical[:,0], s=15, color='red', label='Cold-classical objects')
# ax.scatter(data_hot_classical[:,2],data_hot_classical[:,0], s=15)
# ax.scatter(39.482, 17.16, s=400, label='Pluto', color='indianred')
# ax.scatter(30.11, 17.16, s=400,  label='Neptune', color='lightsteelblue')
ax.set_ylim(0, 10)
ax.set_xlim(41, 49)
ax.set_ylabel(r'Inclination (${\circ}$)')
ax.set_xlabel('Semi-major axis (AU)')
# ax.grid()
ax.legend()

# fig.savefig('./tnos.pdf')