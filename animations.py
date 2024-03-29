import glob, os, rebound
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

sim_name = 'chaos_wide_equalmass_b-3_imp-100_129'
r = '170.0'
b = '3.6'
starting_position = 4.48353
# data = np.array(pd.read_csv(f'./results/{sim_name}_{r}_{b}.csv', index_col=0))
data = np.array(pd.read_csv(f'./thesis_results/{sim_name}.csv', index_col=0))

times = data[:,0]
hash_primary = data[0,2]
m1 = data[0,3]
p = data[:,5:8]
vp = data[:,8:11]
hash_secondary = data[0,11]
m2 = data[0,12]
s = data[:,14:17]
vs = data[:,17:20]
hash_impactor = data[0,20]
mimp = data[0,21]
simp = data[0,22]
imp = data[:,23:26]
vimp = data[:,26:29]

OmegaK = np.sqrt(G*Msun/rsun**3)      # keplerian frequency at this distance
angles = -OmegaK*times                        # one full circle divided up into as many angles as there are outputs

Noutputs = len(data)
dr, dv, mu, h = np.zeros((Noutputs,3)), np.zeros((Noutputs,3)), np.zeros((Noutputs,3)), np.zeros((Noutputs,3))
dr[:,0] = np.linalg.norm(p-s, axis=1)
dr[:,1] = np.linalg.norm(p-imp, axis=1)
dr[:,2] = np.linalg.norm(s-imp, axis=1)
dv[:,0] = np.linalg.norm(vp-vs, axis=1)
dv[:,1] = np.linalg.norm(vp-vimp, axis=1)
dv[:,2] = np.linalg.norm(vs-vimp, axis=1)
h[:,0] = np.cross(p-s,vp-vs)[:,2]
h[:,1] = np.cross(p-imp,vp-vimp)[:,2]
h[:,2] = np.cross(s-imp,vs-vimp)[:,2]
mu[:,0] = G*(m1+m2)
mu[:,1] = G*(m1+mimp)
mu[:,2] = G*(m2+mimp)                           # G times combined mass of secondary and impactor

a = mu*dr/(2*mu-dr*dv**2)                            # semi-major axis between each pair of bodies
energy = -mu/2/a                                    # total energy between each pair of bodies
e = np.sqrt(1+(2*energy*h**2/mu**2))

# plt.figure()
# plt.plot(times,e[:,0])
# plt.ylim(0,1)
Rhill = np.array([rsun*(m1/Msun/3.)**(1./3.), rsun*(m2/Msun/3.)**(1./3.), rsun*(mimp/Msun/3.)**(1./3.)])
Rhill_largest = np.array([np.amax([Rhill[0], Rhill[1]]), np.amax([Rhill[0], Rhill[2]]), np.amax([Rhill[1], Rhill[2]])])
bound = np.logical_and(np.logical_and(energy < 0, np.isfinite(energy)), dr < Rhill_largest)

sp = (s-p)/Rhill[0]         # difference between positions of secondary and primary
ip = (imp-p)/Rhill[0]       # difference between positions of impactor and primary
cosspx, cosspy = np.cos(angles)*sp[:,0], np.cos(angles)*sp[:,1] # cos of reference angles times difference between positions of secondary and primary
sinspx, sinspy = np.sin(angles)*sp[:,0], np.sin(angles)*sp[:,1] # sin of reference angles times difference between positions of secondary and primary

xdot = vs[:,0] - vp[:,0]        # x component of difference in velocities between secondary and primary
ydot = vs[:,1] - vp[:,1]        # y component of difference in velocities between secondary and primary
cosspxdot, cosspydot = np.cos(angles)*xdot, np.cos(angles)*ydot # cos of reference angles times difference between velocities of secondary and primary
sinspxdot, sinspydot = np.sin(angles)*xdot, np.sin(angles)*ydot # sin of reference angles times difference between velocities of secondary and primary

x, y = cosspx-sinspy+rsun, sinspx+cosspy                # x and y values for calculating jacobian constant
vx, vy = cosspxdot-sinspydot, sinspxdot+cosspydot       # vx and vy values for calculating jacobian constant

Cj = n**2*(x**2 + y**2) + 2*(mu[:,0]/dr[:,0] + mu[:,1]/dr[:,1]) - vx**2 - vy**2 # jacobian constant

ref = np.zeros((Noutputs,3))            # reference point that keeps binary at centre of animation
ref[:,0] = 0 + rsun*np.cos(angles)  # x values of reference
ref[:,1] = 0 - rsun*np.sin(angles)      # y values of reference
pref = (p-ref)/Rhill                 # difference between primary and reference point
sref = (s-ref)/Rhill                 # difference between secondary and reference point
impref = (imp-ref)/Rhill             # difference between impactor and reference point
cospx, cospy = np.cos(angles)*pref[:,0], np.cos(angles)*pref[:,1]       # cos of reference angles times relative location of primary
cossx, cossy = np.cos(angles)*sref[:,0], np.cos(angles)*sref[:,1]       # cos of reference angles times relative location of secondary
cosix, cosiy = np.cos(angles)*impref[:,0], np.cos(angles)*impref[:,1]   # cos of reference angles times relative location of impactor
sinpx, sinpy = np.sin(angles)*pref[:,0], np.sin(angles)*pref[:,1]       # sin of reference angles times relative location of primary
sinsx, sinsy = np.sin(angles)*sref[:,0], np.sin(angles)*sref[:,1]       # sin of reference angles times relative location of secondary
sinix, siniy = np.sin(angles)*impref[:,0], np.sin(angles)*impref[:,1]   # sin of reference angles times relative location of impactor

'''2D PLOT OF OUTCOME OF SIMULATION'''
lim = 10
i = -200
fig, axes = plt.subplots(1, figsize=(5,5))
axes.set_xlabel("$x/R_\mathrm{h}$")
axes.set_ylabel("$y/R_\mathrm{h}$")
axes.set_ylim(-lim,lim)
axes.set_xlim(-lim,lim)

primaryhill = plt.Circle((cospx[i]-sinpy[i], sinpx[i]+cospy[i]), Rhill[0]/Rhill[0], fc="none", ec="teal")
secondaryhill = plt.Circle((cossx[i]-sinsy[i], sinsx[i]+cossy[i]), Rhill[1]/Rhill[0], fc="none", ec="hotpink")
impactorhill = plt.Circle((cosix[i]-siniy[i], sinix[i]+cosiy[i]), Rhill[2]/Rhill[0], fc="none", ec="sienna")
axes.add_artist(primaryhill)
axes.add_artist(secondaryhill)
axes.add_artist(impactorhill)

axes.plot(cospx[0:i]-sinpy[0:i], sinpx[0:i]+cospy[0:i], c="teal", lw=1.5)
axes.plot(cossx[0:i]-sinsy[0:i], sinsx[0:i]+cossy[0:i], c="hotpink", lw=1.5)
axes.plot(cosix[0:i]-siniy[0:i], sinix[0:i]+cosiy[0:i], c="sienna", lw=1.5)

axes.plot(cospx[i]-sinpy[i], sinpx[i]+cospy[i], c="teal", marker='o', label="primary")
axes.plot(cossx[i]-sinsy[i], sinsx[i]+cossy[i], c="hotpink", marker='o', label="secondary")
axes.plot(cosix[i]-siniy[i], sinix[i]+cosiy[i], c="sienna", marker='o', label="impactor")
# axes.grid()
axes.legend()
# fig.savefig(f'./img/chaotic_encounters_{sim_name}.pdf', bbox_inches='tight')
# %%
'''2D ANIMATION OF OUTCOME OF SIMULATION'''
lim = 8

color1 = "teal"
color2 = "hotpink"
color3 = "sienna"
lw = 1.5
ms = 8

fig, axes = plt.subplots(1, figsize=(9, 9))
axes.set_xlabel("$x/R_\mathrm{h}$")
axes.set_ylabel("$y/R_\mathrm{h}$")
axes.set_ylim(-lim,lim)
axes.set_xlim(-lim,lim)
primaryline, = axes.plot([], [], label="primary", c=color1, lw=lw)
secondaryline, = axes.plot([], [], label="secondary", c=color2, lw=lw)
impactorline, = axes.plot([], [], label="impactor", c=color3, lw=lw)
primarydot, = axes.plot([], [], marker="o", ms=ms, c=color1)
secondarydot, = axes.plot([], [], marker="o", ms=ms, c=color2)
impactordot, = axes.plot([], [], marker="o", ms=ms, c=color3)
text = axes.text(-lim+(lim/10), lim-(lim/10), '', fontsize=15)
axes.grid()
axes.legend()

primaryhill = plt.Circle((0,0), Rhill[0]/Rhill[0], fc="none", ec=color1) # circle with hill radius of primary - normalized for plotting
secondaryhill = plt.Circle((0,0), Rhill[1]/Rhill[0], fc="none", ec=color2) # circle with hill radius of secondary - normalized for plotting
impactorhill = plt.Circle((0,0), Rhill[2]/Rhill[0], fc="none", ec=color3) # circle with hill radius of impactor - normalized for plotting

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
'''3D ANIMATION OF OUTCOME OF SIMULATION'''
lim = 1
fig = plt.figure(figsize=(9,9))
axes = fig.add_subplot(111, projection='3d')
axes.set_xlabel("$x/R_\mathrm{h}$")
axes.set_ylabel("$y/R_\mathrm{h}$")
axes.set_zlabel("$z/R_\mathrm{h}$")
axes.set_xlim3d([-lim, lim])
axes.set_ylim3d([-lim, lim])
axes.set_zlim3d([-lim, lim])
primaryline, = axes.plot([], [], [], label="primary", c="teal", lw=lw)
secondaryline, = axes.plot([], [], [], label="secondary", c="hotpink", lw=lw)
impactorline, = axes.plot([], [], [], label="impactor", c="sienna", lw=lw)
primarydot, = axes.plot([], [], [], marker="o", ms=ms, c="teal")
secondarydot, = axes.plot([], [], [], marker="o", ms=ms, c="hotpink")
impactordot, = axes.plot([], [], [], marker="o", ms=ms, c="sienna")
text = axes.text(-lim+(lim/10), lim-(lim/10), lim-(lim/10), '', fontsize=15)
axes.legend()

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

anim = animation.FuncAnimation(fig, animate, frames=Noutputs, interval=1)
# %%
x