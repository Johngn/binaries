#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 16:12:22 2020

@author: john
"""

# %%
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

G = 6.67428e-11
au = 1.496e11
rsun = 44.*au
Msun = 1.9891e30

sim_name = 'ecc3'
filenames = glob(f'./rebound/mastersproject/binaries/results/{sim_name}*')

b_all = np.zeros(len(filenames))
simp_all = np.zeros(len(filenames))
e_mean = np.zeros(len(filenames))
e_final = np.zeros(len(filenames))
a_all = np.zeros(len(filenames))
bound = np.zeros((len(filenames), 3))

for i, sim in enumerate(filenames):
    # print(sim)
    data = np.loadtxt(sim)

    times = data[:,0]
    b = data[:,1]
    hash_primary = data[:,2]
    m1 = data[0,3]
    p = data[:,5:8]
    vp = data[:,8:11]
    hash_secondary = data[:,11]
    m2 = data[0,12]
    s = data[:,14:17]
    vs = data[:,17:20]
    hash_impactor = data[:,20]
    mimp = data[0,21]
    simp = data[:,22]/1e3
    imp = data[:,23:26]
    vimp = data[:,26:29]
    
    b_all[i] = b[-1]
    simp_all[i] = simp[-1]

    R, V, mu, h = np.zeros((len(data),3)), np.zeros((len(data),3)), np.zeros((len(data),3)), np.zeros((len(data),3))
    R[:,0] = np.linalg.norm(p-s, axis=1)
    R[:,1] = np.linalg.norm(p-imp, axis=1)
    R[:,2] = np.linalg.norm(s-imp, axis=1)
    V[:,0] = np.linalg.norm(vp-vs, axis=1)
    V[:,1] = np.linalg.norm(vp-vimp, axis=1)
    V[:,2] = np.linalg.norm(vs-vimp, axis=1)
    h[:,0] = np.cross(p-s,vp-vs)[:,2]
    h[:,1] = np.cross(p-imp,vp-vimp)[:,2]
    h[:,2] = np.cross(s-imp,vs-vimp)[:,2]
    mu[:,0] = G*(m1+m2)
    mu[:,1] = G*(m1+mimp)
    mu[:,2] = G*(m2+mimp)

    Rhill = np.array([rsun*(m1/Msun/3.)**(1./3.), rsun*(m2/Msun/3.)**(1./3.), rsun*(mimp/Msun/3.)**(1./3.)])
    Rhill_largest = np.array([np.amax([Rhill[0], Rhill[1]]), np.amax([Rhill[0], Rhill[2]]), np.amax([Rhill[1], Rhill[2]])])
    
    a = mu*R/(2*mu - R*V**2)
    energy = -mu/2/a
    e = np.sqrt(1 + (2*energy*h**2 / mu**2))
    # periapsis = a*(1-e)
    # will_collide = periapsis/3e5 < 1
    
    a_all[i] = a[-1,0]
    
    bound[i] = np.logical_and(np.logical_and(energy < 0, np.isfinite(energy)), R < Rhill_largest)[-1]
    
    e[np.isnan(e)] = 0
    e_mean[i] = np.sqrt(np.mean(e[:,0]**2))
    # print(e_mean[i])
    # print(np.mean(e[:,0]))
    e_final[i] = e[-1,0]
    
    plt.figure()
    plt.plot(times,e)
    plt.ylim(0,1)
    
bound = np.array(bound, dtype='bool')

collisions = glob(f'./rebound/mastersproject/binaries/results/collision_{sim_name}*')
coll_params = np.zeros((len(collisions), 3))
for i, collision in enumerate(collisions):
    params = re.findall('[0-9]+\.[0-9]+', collision)
    params = [float(param) for param in params]
    
    collided_bodies = np.loadtxt(collision)[:,1]
    
    if hash_primary[0] in collided_bodies and hash_secondary[0] in collided_bodies:
        params.append(1)
    if hash_primary[0] in collided_bodies and hash_impactor[0] in collided_bodies:
        params.append(2)
    if hash_secondary[0] in collided_bodies and hash_impactor[0] in collided_bodies:
        params.append(3)
    coll_params[i] = params

plt.figure(figsize=(7,5))
s = 100
plt.scatter(b_all, simp_all, s=1, marker="x", c="black")
plt.scatter(b_all[bound[:,0]], simp_all[bound[:,0]], label='primary-secondary', s=s, c="tab:blue")
plt.scatter(b_all[bound[:,1]], simp_all[bound[:,1]], label='primary-impactor', s=s, c="tab:orange")
plt.scatter(b_all[bound[:,2]], simp_all[bound[:,2]], label='secondary-impactor', s=s, c="tab:green")
plt.scatter(coll_params[coll_params[:,2] == 1][:,1], coll_params[coll_params[:,2] == 1][:,0], marker='x', s=s, c="tab:blue")
plt.scatter(coll_params[coll_params[:,2] == 2][:,1], coll_params[coll_params[:,2] == 2][:,0], marker='x', s=s, c="tab:orange")
plt.scatter(coll_params[coll_params[:,2] == 3][:,1], coll_params[coll_params[:,2] == 3][:,0], marker='x', s=s, c="tab:green")
plt.xlabel("Impact parameter (Hill radii)")
plt.ylabel("Impactor radius (km)")
plt.title(f'{sim_name}')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
plt.yticks(np.asarray((np.unique(simp_all)), dtype=int),)
plt.xticks(np.round(np.unique(b_all), 2))
# plt.xlim(100,350)simp
# plt.savefig(f"./img/{sim_name}_final_bound_whr_dmr", bbox_inches='tight')
# %%
b_all = np.round(b_all, 2)
binary_e = np.ones((len(np.unique(simp_all)), len(np.unique(b_all))))

for i, item in enumerate(np.unique(simp_all)):
    for j, jtem in enumerate(np.unique(b_all)):
        ecc = e_mean[np.logical_and(np.round(simp_all,2) == item, b_all == jtem)]
        if len(ecc > 0):
            binary_e[i,j] = ecc
            
binary_e[binary_e > 1] = np.nan
binary_e = np.round(binary_e, 2)

fig, ax = plt.subplots(1, figsize=(17, 6))
ax = sns.heatmap(binary_e, 
                 annot=True, 
                 linewidths=0.5, 
                 cmap="YlGnBu",
                 yticklabels=np.asarray((np.unique(simp_all)), dtype=int),
                 xticklabels=np.round(np.unique(b_all), 2),
                 # cbar=False
                 )
ax.invert_yaxis()
plt.title(f"Final eccentricity - {sim_name}")
plt.xlabel("Impact parameter (Hill radii)")
plt.ylabel("Impactor radius (km)")
# plt.savefig(f"./img/{sim_name}_final_eccentricity", bbox_inches='tight')
# %%
binary_a = np.ones((len(np.unique(simp_all)), len(np.unique(b_all))))

for i, item in enumerate(np.unique(simp_all)):
    for j, jtem in enumerate(np.unique(b_all)):
        sma = a_all[np.logical_and(np.round(simp_all,2) == item, b_all == jtem)]
        if len(sma > 0):
            binary_a[i,j] = sma
            
binary_a = np.round(binary_a/Rhill[0], 2)
binary_a[binary_a <= 0] = np.nan
binary_a[binary_a > 1] = np.nan

fig, ax = plt.subplots(1, figsize=(17, 6))
ax = sns.heatmap(binary_a, 
                 annot=True, 
                 linewidths=0.5, 
                 cmap="YlGnBu",
                 yticklabels=np.asarray((np.unique(simp_all)), dtype=int),
                 xticklabels=np.round(np.unique(b_all), 2),
                 cbar=False)
ax.invert_yaxis()
plt.title(f"semi-major axis ({sim_name})")
plt.xlabel("Impact parameter (Hill radii)")
plt.ylabel("Impactor radius (km)")
# plt.savefig(f"./img/{sim_name}_final_semimajoraxis", bbox_inches='tight')
# %%
binary_a = np.ones((len(np.unique(simp)), len(np.unique(b))))

for i, item in enumerate(np.unique(simp)):
    for j, jtem in enumerate(np.unique(b)):
        sma = periapsis[np.logical_and(np.round(simp,2) == item, b == jtem)][:,0]
        if len(sma > 0):
            binary_a[i,j] = sma
            
binary_a = np.round(binary_a/Rhill[0], 2)
binary_a[binary_a <= 0] = np.nan
binary_a[binary_a > 10] = np.nan

fig, ax = plt.subplots(1, figsize=(12, 9))
ax = sns.heatmap(binary_a, 
                 annot=True, 
                 linewidths=0.5, 
                 cmap="YlGnBu",
                 yticklabels=np.asarray((np.unique(simp)), dtype=int),
                 xticklabels=np.round(np.unique(b), 2),
                 cbar=False)
ax.invert_yaxis()
plt.title(f"periapsis ({sim_name})")
plt.xlabel("Impact parameter (Hill radii)")
plt.ylabel("Impactor radius (km)")
# plt.savefig(f"./img/{sim_name}_final_periapsis", bbox_inches='tight')