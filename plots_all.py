#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 09:27:19 2020

@author: john
"""

all_files = glob(f'./results/sims/*final*.csv')

results = np.zeros((len(all_files), 3))

for i, file in enumerate(all_files):    
    collided_bodies = pd.read_csv(file)
    coll_params[i] = params
    
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

all_files = glob(f'./results/sims/*final*.csv')

for i, file in enumerate(all_files):
    sim_name = 'OCT20'
    
    data = pd.read_csv(file)
    collisions = glob(f'./results/collision_{sim_name}*')
    coll_params = np.zeros((len(collisions), 3))
    
    for i, collision in enumerate(collisions):
        params = re.findall('[0-9]+\.[0-9]+', collision)
        params = [float(param) for param in params]
        
        collided_bodies = pd.read_csv(collision)['body'].values
        if collided_bodies[0] == 1 and collided_bodies[1] == 2:
            params.append(1)
        if collided_bodies[0] == 1 and collided_bodies[1] == 3:
            params.append(2)
        if collided_bodies[0] == 2 and collided_bodies[1] == 3:
            params.append(3)
        coll_params[i] = params
    
    times = data['time'].to_numpy()
    b = data['b'].to_numpy()
    simp = data['imp radius'].to_numpy()
    m1 = data['mass prim'].to_numpy()
    m2 = data['mass sec'].to_numpy()
    mimp = data['mass imp'].to_numpy()
    p = data[['x prim','y prim', 'z prim']].to_numpy()
    s = data[['x sec','y sec', 'z sec']].to_numpy()
    imp = data[['x imp','y imp', 'z imp']].to_numpy()
    vp = data[['vx prim','vy prim', 'vz prim']].to_numpy()
    vs = data[['vx sec','vy sec', 'vz sec']].to_numpy()
    vimp = data[['vx imp','vy imp', 'vz imp']].to_numpy()
    
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
    
    bound = np.logical_and(np.logical_and(energy < 0, np.isfinite(energy)), R < Rhill_largest)
    # collision = R[:,0] == 0
    
    plt.figure(figsize=(8,6))
    s = 100
    plt.scatter(simp, b, s=1, marker="x", c="black")
    plt.scatter(simp[bound[:,0]], b[bound[:,0]], label='primary-secondary', s=s, c="tab:blue")
    plt.scatter(simp[bound[:,1]], b[bound[:,1]], label='primary-impactor', s=s, c="tab:orange")
    plt.scatter(simp[bound[:,2]], b[bound[:,2]], label='secondary-impactor', s=s, c="tab:green")
    # plt.scatter(simp[collision], b[collision], label='collision', s=s)
    plt.scatter(coll_params[coll_params[:,2] == 1][:,1], coll_params[coll_params[:,2] == 1][:,0], marker='x', s=s*2, c="tab:blue")
    plt.scatter(coll_params[coll_params[:,2] == 2][:,1], coll_params[coll_params[:,2] == 2][:,0], marker='x', s=s*2, c="tab:orange")
    plt.scatter(coll_params[coll_params[:,2] == 3][:,1], coll_params[coll_params[:,2] == 3][:,0], marker='x', s=s*2, c="tab:green")
    plt.xlabel("Impact parameter (Hill radii)")
    plt.ylabel("Impactor radius (km)")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
    # plt.yticks(np.arange(0.5,10.6,0.5))
    # plt.yticks(np.arange(0,101,5))
    # plt.savefig(f"./img/final_bound", bbox_inches='tight')