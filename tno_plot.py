#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 18:48:29 2020

@author: John Gillan

This takes all known Kuiper belt objects and makes a plot of semi-major axis vs inclination
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %%
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

classical = np.logical_and(data_clean[:,2] > 41, data_clean[:,2] < 48.7) # select objects with range of 'a'
cold = data_clean[:,0] < 5 # low inclination
hot = data_clean[:,0] > 5 # high inclination

cold_classical = np.logical_and(cold, classical)
data_cold_classical = data_clean[cold_classical]

hot_classical = np.logical_and(hot, classical)
data_hot_classical = data_clean[hot_classical]

df = pd.DataFrame(data_clean, columns=('i','e','a'))
df.to_csv('./data/tnos_clean', index=None)
# %%
data_clean = pd.read_csv('./data/tnos_clean')

fig, ax = plt.subplots(2, figsize=(7,7))

ax[0].scatter(data_clean['a'],data_clean['i'], s=7, alpha=0.7)
ax[0].scatter(data_cold_classical[:,2],data_cold_classical[:,0], s=7, color='red', label='Cold-classical objects')
ax[0].scatter(data_hot_classical[:,2],data_hot_classical[:,0], s=15)
ax[0].scatter(39.482, 17.16, s=400, label='Pluto', color='indianred')
ax[0].scatter(30.11, 1.77, s=400,  label='Neptune', color ='lightsteelblue')
ax[0].set_ylim(0, 40)
ax[0].set_xlim(35, 55)
ax[0].set_yticks(np.arange(0,41,5))
ax[0].axvline(44)
ax[0].set_ylabel(r'Inclination [$^\circ$]')
ax[0].set_xlabel('Semi-major axis [AU]')

ax[1].scatter(data_clean['a'],data_clean['i'], s=7, alpha=0.7)
ax[1].scatter(data_cold_classical[:,2],data_cold_classical[:,0], s=7, color='red', label='Cold-classical objects')
ax[1].scatter(data_hot_classical[:,2],data_hot_classical[:,0], s=7)
ax[1].scatter(30.11, 1.77, s=400,  label='Neptune', color ='lightsteelblue')
ax[1].set_ylim(0, 40)
ax[1].set_xlim(35, 55)
ax[1].set_yticks(np.arange(0,41,5))
ax[1].axvline(44)
ax[1].set_ylabel(r'Inclination [$^\circ$]')
ax[1].set_xlabel('Semi-major axis [AU]')
# ax.grid()
# ax.legend()

# fig.savefig('./cckbos.pdf', bbox_inches='tight')