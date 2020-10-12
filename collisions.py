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

sim_name = 'COLL'
b = '0.3'
r = '1001.0'

coll_data = pd.read_csv(f'./results/collision_{sim_name}__b-{b}__r-{r}.csv')
data = pd.read_csv(f'./results/{sim_name}__b-{b}__r-{r}.csv')
noutputs = len(data)

times = data['time'].to_numpy()
p = data[['x prim','y prim','z prim']].to_numpy()
s = data[['x sec','y sec','z sec']].to_numpy()
imp = data[['x imp','y imp','z imp']].to_numpy()
vp = data[['vx prim','vy prim','vz prim']].to_numpy()
vs = data[['vx sec','vy sec','vz sec']].to_numpy()
vimp = data[['vx imp','vy imp','vz imp']].to_numpy()

coll_time = coll_data['time'].to_numpy()
bodies = coll_data['body'].to_numpy()
radius = coll_data['r'].to_numpy()
m = coll_data['m'].to_numpy()
r = coll_data[['x','y','z']].to_numpy()
v = coll_data[['vx','vy','vz']].to_numpy()

g = 6.67428e-11                             # gravitational constanct in SI units
au = 1.496e11                               # astronomical unit    
msun = 1.9891e30                            # mass of sun
rsun = 44.*au                               # distance of centre of mass of binary from the sun 
omegaK = np.sqrt(g*msun/rsun**3)            # keplerian frequency at this distance
rhill = rsun*(m/msun/3.)**(1./3.)        # Hill radius of binary
vk = np.sqrt(g*msun/rsun)                   # keplerian velocity at this distance
angles = -omegaK*times                      # angles of reference point at each time


ref = np.zeros((noutputs,3))            # reference point that keeps binary at centre of animation
ref[:,0] = 0 + rsun*np.cos(angles)      # x values of reference
ref[:,1] = 0 - rsun*np.sin(angles)      # y values of reference

# v_ref = np.zeros((noutputs,3))
# v_ref[:,0] = np.sin(angles)                 # x values of azimuthal unit vector (reference point)
# v_ref[:,1] = np.cos(angles)                 # y values of azimuthal unit vector (reference point)
# va = vk*v_ref                               # azimuthal velocity vector (reference point)

theta = -omegaK*coll_time[0]                # angle at which collision occured
vref = np.array([np.sin(theta),np.cos(theta),0])*vk

v1 = v[0]-vref                                # velocity of body 1 in reference frame
v2 = v[1]-vref                                # velocity of body 2 in reference frame
u_1 = v1/np.linalg.norm(v1)                 # unit velocity vector of body 1 in reference frame
u_2 = v2/np.linalg.norm(v2)                 # unit velocity vector of body 2 in reference frame

v_pos = r[0]-r[1]
v_dir = v[0]-v[1]
dr = np.linalg.norm(v_pos)              # distance between bodies
dv = np.linalg.norm(v_dir)

a = v_pos
n = v_dir/dv

b = np.linalg.norm(a-np.dot(a,n)*n)

collision_angle = np.arccos(np.dot(v[0],v[1])/np.dot(np.linalg.norm(v[0]),np.linalg.norm(v[1])))
collision_angle = np.arccos(np.dot(v1,v2)/np.dot(np.linalg.norm(v1),np.linalg.norm(v2)))
collision_angle_deg = np.rad2deg(collision_angle)
