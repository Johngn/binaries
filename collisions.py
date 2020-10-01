#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 10:51:31 2020

@author: johngillan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sim_name = 'coord_test3'
b = '2.4'
r = '160.0'

coll_data = pd.read_csv(f'./results/collision_{sim_name}__b-{b}__r-{r}.csv')
data = pd.read_csv(f'./results/{sim_name}__b-{b}__r-{r}.csv')
Noutputs = len(data)

times = data['time'].to_numpy()
p = data[['x prim','y prim','z prim']].to_numpy()
s = data[['x sec','y sec','z sec']].to_numpy()
imp = data[['x imp','y imp','z imp']].to_numpy()
vp = data[['vx prim','vy prim','vz prim']].to_numpy()
vs = data[['vx sec','vy sec','vz sec']].to_numpy()
vimp = data[['vx imp','vy imp','vz imp']].to_numpy()

coll_time = coll_data['time'].to_numpy()
bodies = coll_data['body'].to_numpy()
r = coll_data['r'].to_numpy()
m = coll_data['m'].to_numpy()
R = coll_data[['x','y','z']].to_numpy()
V = coll_data[['vx','vy','vz']].to_numpy()

G = 6.67428e-11                             # gravitational constanct in SI units
au = 1.496e11                               # astronomical unit    
Msun = 1.9891e30                            # mass of sun
rsun = 44.*au                                  # distance of centre of mass of binary from the sun 
OmegaK = np.sqrt(G*Msun/rsun**3)       # keplerian frequency at this distance
vk = np.sqrt(G*Msun/rsun)
angles = -OmegaK*times
theta = -OmegaK*coll_time[0]
v_ref = np.zeros((Noutputs,3))
v_ref[:,0] = np.sin(angles)
v_ref[:,1] = np.cos(angles)
va = vk*v_ref



V1 = V[0]-va
V2 = V[1]-va
u_1 = V1/np.linalg.norm(V1)
u_2 = V2/np.linalg.norm(V2)
dr = np.linalg.norm(R[0]-R[1])
dv = np.linalg.norm(V[0]-V[1])
dv2 = np.linalg.norm(v1-v2)

ref = np.zeros((Noutputs,3))            # reference point that keeps binary at centre of animation
ref[:,0] = 0 + rsun*np.cos(angles)  # x values of reference
ref[:,1] = 0 - rsun*np.sin(angles)      # y values of reference



# ref = np.array([0 + rsun*np.cos(theta),0 - rsun*np.sin(theta)])

collision_angle = np.arccos(np.dot(V[0],V[1])/np.dot(np.linalg.norm(V[0]),np.linalg.norm(V[1])))
collision_angle = np.arccos(np.dot(v1,v2)/np.dot(np.linalg.norm(v1),np.linalg.norm(v2)))
collision_angle_deg = np.rad2deg(collision_angle)
