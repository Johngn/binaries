#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 10:06:39 2020

@author: john
"""

import rebound
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation
from matplotlib.animation import FuncAnimation

g = 6.67428e-11                             # gravitational constanct in SI units
m1 = 1e3                # mass of primary calculated from density and radius
m2 = 0
rbin = 1                            # separation of binary
e = 0.99
r_a = rbin*(1-e)

xb1 = -m2/(m1+m2)*r_a
xb2 = m1/(m1+m2)*r_a

vorb = np.sqrt(g*(m1+m2)*(2/r_a-1/rbin))
vorb1 = -m2/(m1+m2)*vorb
vorb2 = m1/(m1+m2)*vorb

def setupSimulation():
    sim = rebound.Simulation()              # initialize rebound simulation
    sim.G = g                               # set G which sets units of integrator - SI in this case
    sim.add(m=m1, x=xb1, vy=vorb1, hash="primary")
    sim.add(m=m2, x=xb2, vy=vorb2, hash="secondary")
    # sim.add(m=)
    return sim

sim = setupSimulation()

noutputs = 1000             # number of outputs
p, s = np.zeros((noutputs, 3)), np.zeros((noutputs, 3)) # position
vp, vs = np.zeros((noutputs, 3)), np.zeros((noutputs, 3)) # velocity
totaltime = 100000    
times = np.linspace(0.,totaltime, noutputs) # create times for integrations
ps = sim.particles                      # create variable containing particles in simulation

for k, time in enumerate(times):
    sim.integrate(time)
    p[k] = [ps["primary"].x, ps["primary"].y, ps["primary"].z]
    s[k] = [ps["secondary"].x, ps["secondary"].y, ps["secondary"].z]
    vp[k] = [ps["primary"].vx, ps["primary"].vy, ps["primary"].vz]
    vs[k] = [ps["secondary"].vx, ps["secondary"].vy, ps["secondary"].vz]

dr = np.linalg.norm(p-s, axis=1)                # distance between primary and secondary
dv = np.linalg.norm(vp-vs, axis=1)              # relative velocity between primary and secondary
mu = g*(m1+m2)                                 # G times combined mass of primary and secondary
h = np.cross(p-s,vp-vs)[:,2]                       # angular momentum

semimajoraxis = mu*dr/(2*mu-dr*dv**2)
energy = -mu/2/semimajoraxis
ecc = np.sqrt(1+(2*energy*h**2/mu**2))

lim = 2
fig, axes = plt.subplots(1, figsize=(9, 9))
axes.set_ylim(-lim,lim)
axes.set_xlim(-lim,lim)
primaryline, = axes.plot([], [], label="primary", c="tab:orange", lw=1.2)
secondaryline, = axes.plot([], [], label="secondary", c="tab:blue", lw=1.2)
primarydot, = axes.plot([], [], marker="o", ms=7, c="tab:orange")   
secondarydot, = axes.plot([], [], marker="o", ms=7, c="tab:blue")
axes.grid()

def animate(i):
    primaryline.set_data(p[0:i,0], p[0:i,1])
    secondaryline.set_data(s[0:i,0], s[0:i,1])
    primarydot.set_data(p[i,0], p[i,1])
    secondarydot.set_data(s[i,0], s[i,1])
    return primarydot, secondarydot, primaryline, secondaryline

anim = animation.FuncAnimation(fig, animate, frames=noutputs, interval=1, blit=True)