# %%
import glob, os, csv, rebound
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from timeit import default_timer as timed
from matplotlib import animation
from matplotlib.animation import FuncAnimation, FFMpegWriter

g = 6.67428e-11                             # gravitational constanct in SI units
au = 1.496e11                               # astronomical unit    
msun = 1.9891e30                            # mass of sun
m1 = 1e18                # mass of primary calculated from density and radius
m2 = 0               # mass of secondary calculated from density and radius
rsun = 44.*au                                  # distance of centre of mass of binary from the sun 
rhill1 = rsun*(m1/msun/3.)**(1./3.)        # Hill radius of primary
rbin = 0.5*rhill1                            # separation of binary
vorb = np.sqrt(g*(m1+m2)/rbin)              # orbital speed of primary and secondary around each other
t = 2.*np.pi/np.sqrt(g*msun/rsun**3)         # orbital period of binary around the sun
vk = np.sqrt(g*msun/rsun)      # orbital speed of binary around sun

e = 0
r_a = rbin*(1-e)

xb1 = -m2/(m1+m2)*r_a                  # slightly adjust initial x position of primary to keep centre of mass of binary at r
xb2 = m1/(m1+m2)*r_a                   # slightly adjust initial x position of secondary to keep centre of mass of binary at r

vorb = np.sqrt(g*(m1+m2)*(2/r_a-1/rbin))
vorb1 = -m2/(m1+m2)*vorb                # orbital speed of primary around secondary - adjusted to account for offset from COM
vorb2 = m1/(m1+m2)*vorb                 # orbital speed of secondary around primary - adjusted to account for offset from COM

def setupSimulation():
    sim = rebound.Simulation()              # initialize rebound simulation
    sim.G = g                               # set G which sets units of integrator - SI in this case
    sim.collision = 'direct'
    # sim.add(m=msun, hash="sun")
    sim.add(m=m1, x=rsun+xb1, vy=vk+vorb1, hash="primary")
    sim.add(m=m2, x=rsun+xb2, vy=vk+vorb2, hash="secondary")
    return sim

sim = setupSimulation()

noutputs = 1000             # number of outputs
p, s = np.zeros((noutputs, 3)), np.zeros((noutputs, 3))# position
vp, vs = np.zeros((noutputs, 3)), np.zeros((noutputs, 3)) # velocity
totaltime = t*1
times = np.linspace(0.,totaltime, noutputs) # create times for integrations
ps = sim.particles                      # create variable containing particles in simulation

all_ps = [p.hash.value for j, p in enumerate(ps)]

timer = timed() # start timer to time simulations

for k, time in enumerate(times):
    sim.integrate(time)
    # print(k)
    p[k] = [ps["primary"].x, ps["primary"].y, ps["primary"].z]
    s[k] = [ps["secondary"].x, ps["secondary"].y, ps["secondary"].z]
    vp[k] = [ps["primary"].vx, ps["primary"].vy, ps["primary"].vz]
    vs[k] = [ps["secondary"].vx, ps["secondary"].vy, ps["secondary"].vz]

print(timed()-timer) # finish timer

lim = 1
fig, axes = plt.subplots(1, figsize=(7, 7))
axes.set_xlabel("$x/R_\mathrm{h}$")
axes.set_ylabel("$y/R_\mathrm{h}$")
axes.set_ylim(-lim,lim)
axes.set_xlim(-lim,lim)
primaryline, = axes.plot([], [], label="primary", c="tab:orange", lw=1.2)
secondaryline, = axes.plot([], [], label="secondary", c="tab:blue", lw=1.2)
primarydot, = axes.plot([], [], marker="o", ms=7, c="tab:orange")   
secondarydot, = axes.plot([], [], marker="o", ms=7, c="tab:blue")
text = axes.text(-lim+(lim/10), lim-(lim/10), '', fontsize=15)
axes.legend()
axes.grid()

omegak = np.sqrt(g*msun/rsun**3)       # keplerian frequency at this distance
angles = -omegak*times
ref = np.zeros((noutputs,3))
ref[:,0] = rsun*np.cos(angles)
ref[:,1] = 0 - rsun*np.sin(angles)

pref = (p-ref)/rhill1
sref = (s-ref)/rhill1
cospx, cospy = np.cos(angles)*pref[:,0], np.cos(angles)*pref[:,1]
cossx, cossy = np.cos(angles)*sref[:,0], np.cos(angles)*sref[:,1]
sinpx, sinpy = np.sin(angles)*pref[:,0], np.sin(angles)*pref[:,1]
sinsx, sinsy = np.sin(angles)*sref[:,0], np.sin(angles)*sref[:,1]

def animate(i):    
    primaryline.set_data(cospx[0:i]-sinpy[0:i], sinpx[0:i]+cospy[0:i])
    secondaryline.set_data(cossx[0:i]-sinsy[0:i], sinsx[0:i]+cossy[0:i])
    primarydot.set_data(cospx[i]-sinpy[i], sinpx[i]+cospy[i])
    secondarydot.set_data(cossx[i]-sinsy[i], sinsx[i]+cossy[i])
    return primarydot, secondarydot, primaryline, secondaryline, text

anim = animation.FuncAnimation(fig, animate, frames=noutputs, interval=1, blit=True)

