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

sim_name = 'OCT24'
b = '2.0'
r = '1500.0'

coll_data = pd.read_csv(f'./results/collisions/collision_{sim_name}_b-{b}_r-{r}.csv')
bodies = coll_data['body'].to_numpy()
radius = coll_data['r'].to_numpy()
m = coll_data['m'].to_numpy()
r = coll_data[['x','y','z']].to_numpy()
v = coll_data[['vx','vy','vz']].to_numpy()

g = 6.67428e-11                             # gravitational constanct in SI units

position_vector = r[0]-r[1]
velocity_vector = v[0]-v[1]
dr = np.linalg.norm(position_vector)              # distance between bodies
dv = np.linalg.norm(velocity_vector)              # collision speed

position_unit_vector = position_vector/np.linalg.norm(position_vector)

collision_unit_vector = -position_vector/dr
collision_speed = np.dot(velocity_vector,collision_unit_vector)

n = velocity_vector/dv

B = np.linalg.norm(position_vector-np.dot(position_vector,n)*n)
theta = np.arcsin(B/dr)
# theta = np.rad2deg(theta)

M_tot = m[0]+m[1]
mu = m[0]*m[1]/M_tot
v_cm = (m[0]*v[0] + m[1]*v[1])/(m[0]+m[1])
v_p = np.linalg.norm(v[0]-v_cm)
v_targ = np.linalg.norm(v[1]-v_cm)
q_r = (0.5*m[0]*v_p**2 + 0.5*m[1]*v_targ**2)/M_tot
q_r = 0.5*mu*dv**2/M_tot

gravitational_binding_energy = 3*g*m/(5*radius)

b = B/dr

b_crit = np.amax(radius)/dr

escape_speed = np.sqrt(2*g*m/radius)     # escape speed for each body
fragmentation = collision_speed > escape_speed        # collision causes fragmentation if speed greater than escape speed


# %%
data = pd.read_csv(f'./results/{sim_name}_b-{b}_r-{r}.csv')
noutputs = len(data)
times = data['time'].to_numpy()
coll_time = coll_data['time'].to_numpy()
au = 1.496e11                               # astronomical unit
msun = 1.9891e30                            # mass of sun
rsun = 44.*au                               # distance of centre of mass of binary from the sun
omegak = np.sqrt(g*msun/rsun**3)            # keplerian frequency at this distance
vk = np.sqrt(g*msun/rsun)                   # keplerian velocity at this distance
angles = -omegak*times                      # angles of reference point at each time
theta = -omegaK*coll_time[0]                # angle at which collision occured
vref = np.array([np.sin(theta),np.cos(theta),0])*vk

v1 = v[0]-vref                                # velocity of body 1 in reference frame
v2 = v[1]-vref                                # velocity of body 2 in reference frame
u_1 = v1/np.linalg.norm(v1)                 # unit velocity vector of body 1 in reference frame
u_2 = v2/np.linalg.norm(v2)                 # unit velocity vector of body 2 in reference frame

collision_angle = np.arccos(np.dot(v1,v2)/np.dot(np.linalg.norm(v1),np.linalg.norm(v2)))
collision_angle_deg = np.rad2deg(collision_angle)

ref = np.zeros((noutputs,3))            # reference point that keeps binary at centre of animation
ref[:,0] = 0 + rsun*np.cos(angles)      # x values of reference
ref[:,1] = 0 - rsun*np.sin(angles)      # y values of reference
v_ref = np.zeros((noutputs,3))
v_ref[:,0] = np.sin(angles)                 # x values of azimuthal unit vector (reference point)
v_ref[:,1] = np.cos(angles)                 # y values of azimuthal unit vector (reference point)
va = vk*v_ref                               # azimuthal velocity vector (reference point)