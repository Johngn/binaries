# %%
import glob, os, csv, rebound
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation
from timeit import default_timer as timed
from sqlalchemy import create_engine
from matplotlib.animation import FuncAnimation, FFMpegWriter

G = 6.67428e-11                             # gravitational constanct in SI units
au = 1.496e11                               # astronomical unit    
Msun = 1.9891e30                            # mass of sun
rsun = 44.*au                               # distance of centre of mass of binary from the sun 
T = 2.*np.pi/np.sqrt(G*(Msun)/rsun**3)      # orbital period of binary around the sun
n = 2*np.pi/T                               # mean motion of binary around the sun
year = 365.25*24.*60.*60.                   # number of seconds in a year
Noutputs = 1000                             # number of outputs for plotting

# read data from csv into dataframe
# data = pd.read_csv(f'./results/particles__b-2.4999999999999998__r-5.0.csv')
data = pd.read_csv(f'./results/presentation__b-4.5__r-30.0.csv')

# create numpy arrays from dataframe
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

# empty arrays for values
R, V, Rhill, mu, h, e = np.zeros((len(data),3)), np.zeros((len(data),3)), np.zeros((len(data),3)), np.zeros((len(data),3)),np.zeros((len(data),3)),np.zeros((len(data),3))

R[:,0] = np.linalg.norm(p-s, axis=1)                # distance between primary and secondary
R[:,1] = np.linalg.norm(p-imp, axis=1)              # distance between primary and impactor
R[:,2] = np.linalg.norm(s-imp, axis=1)              # distance between secondary and impactor

V[:,0] = np.linalg.norm(vp-vs, axis=1)              # relative velocity between primary and secondary
V[:,1] = np.linalg.norm(vp-vimp, axis=1)            # relative velocity between primary and impactor
V[:,2] = np.linalg.norm(vs-vimp, axis=1)            # relative velocity between secondary and impactor

Rhill[:,0] = rsun*((m1+m2)/Msun/3.)**(1./3.)        # combined Hill radius of primary and secondary
Rhill[:,1] = rsun*((m1+mimp)/Msun/3.)**(1./3.)      # combined Hill radius of primary and impactor
Rhill[:,2] = rsun*((m2+mimp)/Msun/3.)**(1./3.)      # combined Hill radius of secondary and impactor

mu[:,0] = G*(m1+m2)                                 # G times combined mass of primary and secondary
mu[:,1] = G*(m1+mimp)                               # G times combined mass of primary and impactor
mu[:,2] = G*(m2+mimp)                               # G times combined mass of secondary and impactor

a = mu*R/(2*mu - R*V**2)                            # semi-major axis between each pair of bodies
energy = -mu/2/a                                    # total energy between each pair of bodies
bound = np.logical_and(energy < 0, R < Rhill)       # bodies are bound if their energy is less than zero and they are closer together than the Hill radius

distance1 = p-s                                     # difference between x, y and z values of primary and secondary
distance2 = p-imp                                   # difference between x, y and z values of primary and impactor
distance3 = s-imp                                   # difference between x, y and z values of secondary and impactor
v1 = vp-vs                                          # difference between vx, vy and vz values of primary and secondary
v2 = vp-vimp                                        # difference between vx, vy and vz values of primary and impactor
v3 = vs-vimp                                        # difference between vx, vy and vz values of secondary and impactor
h[:,0] = np.cross(distance1,v1)[:,2]
h[:,1] = np.cross(distance2,v2)[:,2]
h[:,2] = np.cross(distance3,v3)[:,2]
e = np.sqrt(1 + (2 * energy * h**2 / mu**2))

OmegaK = np.sqrt(G*(Msun+m1[0]+m2[0])/rsun**3)      # keplerian frequency at this distance
angles = -OmegaK*times

sp = (s-p)/Rhill[0,0]
ip = (imp-p)/Rhill[0,1]
cosspx, cosspy = np.cos(angles)*sp[:,0], np.cos(angles)*sp[:,1]
sinspx, sinspy = np.sin(angles)*sp[:,0], np.sin(angles)*sp[:,1]

xdot = vs[:,0] - vp[:,0]
ydot = vs[:,1] - vp[:,1]
cosspxdot, cosspydot = np.cos(angles)*xdot, np.cos(angles)*ydot
sinspxdot, sinspydot = np.sin(angles)*xdot, np.sin(angles)*ydot

x, y = cosspx-sinspy+rsun, sinspx+cosspy
vx, vy = cosspxdot-sinspydot, sinspxdot+cosspydot

Cj = n**2*(x**2 + y**2) + 2*(mu[:,0]/R[:,0] + mu[:,1]/R[:,1]) - vx**2 - vy**2 # jacobian constant
# %%
lim = 5
fig, axes = plt.subplots(1, figsize=(9, 9))
axes.set_xlabel("$x/R_\mathrm{h}$")
axes.set_ylabel("$y/R_\mathrm{h}$")
axes.set_ylim(-lim,lim)
axes.set_xlim(-lim,lim)
primaryline, = axes.plot([], [], label="primary", c="tab:orange", lw=1.2)
secondaryline, = axes.plot([], [], label="secondary", c="tab:blue", lw=1.2)
impactorline, = axes.plot([], [], label="impactor", c="tab:green", lw=1.2)
primarydot, = axes.plot([], [], marker="o", ms=7, c="tab:orange")
secondarydot, = axes.plot([], [], marker="o", ms=7, c="tab:blue")
impactordot, = axes.plot([], [], marker="o", ms=7, c="tab:green")
text = axes.text(-lim+(lim/10), lim-(lim/10), '', fontsize=15)
# axes.grid()
axes.legend()

ref = np.zeros((Noutputs,3))
ref[:,0] = -rsun + rsun*np.cos(angles)
ref[:,1] = 0 - rsun*np.sin(angles)

pref = (p-ref)/Rhill[0,0]
sref = (s-ref)/Rhill[0,0]
impref = (imp-ref)/Rhill[0,0]
cospx, cospy = np.cos(angles)*pref[:,0], np.cos(angles)*pref[:,1]
cossx, cossy = np.cos(angles)*sref[:,0], np.cos(angles)*sref[:,1]
cosix, cosiy = np.cos(angles)*impref[:,0], np.cos(angles)*impref[:,1]
sinpx, sinpy = np.sin(angles)*pref[:,0], np.sin(angles)*pref[:,1]
sinsx, sinsy = np.sin(angles)*sref[:,0], np.sin(angles)*sref[:,1]
sinix, siniy = np.sin(angles)*impref[:,0], np.sin(angles)*impref[:,1]

Rhillprim = rsun*(m1/Msun/3.)**(1./3.)/Rhill[0,0]
Rhillsec = rsun*(m2/Msun/3.)**(1./3.)/Rhill[0,0]
Rhillimp = rsun*(mimp/Msun/3.)**(1./3.)/Rhill[0,0]
primaryhill = plt.Circle((0,0), Rhillprim[0], fc="none", ec="tab:orange")
secondaryhill = plt.Circle((0,0), Rhillsec[0], fc="none", ec="tab:blue")
impactorhill = plt.Circle((0,0), Rhillimp[0], fc="none", ec="tab:green")

def init():
    axes.add_patch(primaryhill)
    axes.add_patch(secondaryhill)
    axes.add_patch(impactorhill)
    return primaryhill, secondaryhill, impactorhill,

def animate(i):
    primaryline.set_data(cospx[0:i]-sinpy[0:i], sinpx[0:i]+cospy[0:i])
    secondaryline.set_data(cossx[0:i]-sinsy[0:i], sinsx[0:i]+cossy[0:i])
    impactorline.set_data(cosix[0:i]-siniy[0:i], sinix[0:i]+cosiy[0:i])
    primarydot.set_data(cospx[i]-sinpy[i], sinpx[i]+cospy[i])
    secondarydot.set_data(cossx[i]-sinsy[i], sinsx[i]+cossy[i])
    impactordot.set_data(cosix[i]-siniy[i], sinix[i]+cosiy[i])    
    primaryhill.center = (cospx[i]-sinpy[i], sinpx[i]+cospy[i])
    secondaryhill.center = (cossx[i]-sinsy[i], sinsx[i]+cossy[i])
    impactorhill.center = (cosix[i]-siniy[i], sinix[i]+cosiy[i])
    text.set_text('{} Years'.format(int(times[i]/(year))))
    return primarydot, secondarydot, impactordot, primaryline, secondaryline, impactorline, text, primaryhill, secondaryhill, impactorhill

anim = animation.FuncAnimation(fig, animate, init_func=init,  frames=Noutputs, interval=1, blit=True)
# %%
plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

f = f'vid/ps_animation.mp4' 
writervideo = FFMpegWriter(fps=10) # ffmpeg must be installed
anim.save(f, writer=writervideo)
# %%
lim = 10
fig = plt.figure(figsize=(12,12))
axes = fig.add_subplot(111, projection='3d')
axes.set_xlabel("$x/R_\mathrm{h}$")
axes.set_ylabel("$y/R_\mathrm{h}$")
axes.set_zlabel("$z/R_\mathrm{h}$")
axes.set_xlim3d([-lim, lim])
axes.set_ylim3d([-lim, lim])
axes.set_zlim3d([-lim, lim])
primaryline, = axes.plot([], [], [], label="primary", c="tab:orange", lw=1.5)
secondaryline, = axes.plot([], [], [], label="secondary", c="tab:blue", lw=1.5)
impactorline, = axes.plot([], [], [], label="impactor", c="tab:green", lw=1.5)
primarydot, = axes.plot([], [], [], marker="o", ms=8, c="tab:orange")
secondarydot, = axes.plot([], [], [], marker="o", ms=8, c="tab:blue")
impactordot, = axes.plot([], [], [], marker="o", ms=8, c="tab:green")
text = axes.text(-lim+(lim/10), lim-(lim/10), lim-(lim/10), '', fontsize=15)
axes.legend()

angles = -OmegaK*times
cosspx, cosspy = np.cos(angles)*sp[:,0], np.cos(angles)*sp[:,1]
cosipx, cosipy = np.cos(angles)*ip[:,0], np.cos(angles)*ip[:,1]
sinspx, sinspy = np.sin(angles)*sp[:,0], np.sin(angles)*sp[:,1]
sinipx, sinipy = np.sin(angles)*ip[:,0], np.sin(angles)*ip[:,1]

def animate(i):
    primaryline.set_data(cospx[0:i]-sinpy[0:i], sinpx[0:i]+cospy[0:i])
    secondaryline.set_data(cossx[0:i]-sinsy[0:i], sinsx[0:i]+cossy[0:i])
    impactorline.set_data(cosix[0:i]-siniy[0:i], sinix[0:i]+cosiy[0:i])
    primaryline.set_3d_properties(pref[0:i,2])
    secondaryline.set_3d_properties(sref[0:i,2])
    impactorline.set_3d_properties(impref[0:i,2])
    primarydot.set_data(cospx[i]-sinpy[i], sinpx[i]+cospy[i])
    secondarydot.set_data(cossx[i]-sinsy[i], sinsx[i]+cossy[i])
    impactordot.set_data(cosix[i]-siniy[i], sinix[i]+cosiy[i])
    primarydot.set_3d_properties(pref[i,2])
    secondarydot.set_3d_properties(sref[i,2])
    impactordot.set_3d_properties(impref[i,2])
    text.set_text('{} Years'.format(int(times[i]/(year))))
    return primarydot, secondarydot, impactordot, primaryline, secondaryline, impactorline, text

anim = animation.FuncAnimation(fig, animate, frames=len(data), interval=1)
# anim.save(f'{path}/videos/3D.mp4')
# %%
lim = 10
fig, axes = plt.subplots(1, figsize=(5, 5))
axes.set_xlabel("$x/R_\mathrm{h}$")
axes.set_ylabel("$y/R_\mathrm{h}$")
axes.set_ylim(-lim,lim)
axes.set_xlim(-lim,lim)

ref = np.zeros((Noutputs,3))
ref[:,0] = -rsun + rsun*np.cos(angles)
ref[:,1] = 0 - rsun*np.sin(angles)

pref = (p-ref)/Rhill[0,0]
sref = (s-ref)/Rhill[0,0]
impref = (imp-ref)/Rhill[0,0]
cospx, cospy = np.cos(angles)*pref[:,0], np.cos(angles)*pref[:,1]
cossx, cossy = np.cos(angles)*sref[:,0], np.cos(angles)*sref[:,1]
cosix, cosiy = np.cos(angles)*impref[:,0], np.cos(angles)*impref[:,1]
sinpx, sinpy = np.sin(angles)*pref[:,0], np.sin(angles)*pref[:,1]
sinsx, sinsy = np.sin(angles)*sref[:,0], np.sin(angles)*sref[:,1]
sinix, siniy = np.sin(angles)*impref[:,0], np.sin(angles)*impref[:,1]

Rhillprim = rsun*(m1/Msun/3.)**(1./3.)/Rhill[0,0]
Rhillsec = rsun*(m2/Msun/3.)**(1./3.)/Rhill[0,0]
Rhillimp = rsun*(mimp/Msun/3.)**(1./3.)/Rhill[0,0]
primaryhill = plt.Circle((cospx[-1]-sinpy[-1], sinpx[-1]+cospy[-1]), Rhillprim[0], fc="none", ec="tab:orange")
axes.add_artist(primaryhill)
secondaryhill = plt.Circle((cossx[-1]-sinsy[-1], sinsx[-1]+cossy[-1]), Rhillsec[0], fc="none", ec="tab:blue")
axes.add_artist(secondaryhill)
impactorhill = plt.Circle((cosix[-1]-siniy[-1], sinix[-1]+cosiy[-1]), Rhillimp[0], fc="none", ec="tab:green")
axes.add_artist(impactorhill)

axes.plot(cospx-sinpy, sinpx+cospy, label="primary", c="tab:orange", lw=1.5)
axes.plot(cossx-sinsy, sinsx+cossy, label="secondary", c="tab:blue", lw=1.5)
axes.plot(cosix-siniy, sinix+cosiy, label="impactor", c="tab:green", lw=1.5)

axes.plot(cospx[-1]-sinpy[-1], sinpx[-1]+cospy[-1], c="tab:orange", marker='o')
axes.plot(cossx[-1]-sinsy[-1], sinsx[-1]+cossy[-1], c="tab:blue", marker='o')
axes.plot(cosix[-1]-siniy[-1], sinix[-1]+cosiy[-1], c="tab:green", marker='o')

axes.legend()
fig.savefig('./result5.pdf', bbox_inches='tight')