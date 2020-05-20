# %%
import csv
import numpy as np
import rebound
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation
from scipy.spatial.distance import pdist
from timeit import default_timer as timed
# %%
b = 3.5
simp = 5.0
path = '/home/john/Desktop/mastersproject/results'

data = np.loadtxt(f'{path}/particles__b-{b}__r-{simp}.csv', delimiter=',')

times = data[:,0]
m1 = data[0,1]
m2 = data[0,8]
mimp = data[0,15]
Msun = data[0,22]
p = data[:,2:5]
s = data[:,9:12]
imp = data[:,16:19]
sun = data[:,23:26]
vp = data[:,5:8]
vs = data[:,12:15]
vimp = data[:,19:22]
vsun = data[:,26:29]

r1 = np.linalg.norm(p-s, axis=1)
r2 = np.linalg.norm(p-imp, axis=1)
r3 = np.linalg.norm(s-imp, axis=1)
v1 = np.linalg.norm(vp-vs, axis=1)
v2 = np.linalg.norm(vp-vimp, axis=1)
v3 = np.linalg.norm(vs-vimp, axis=1)

G = 6.67428e-11
au = 1.496e11
r = 44.*au
T = 2.*np.pi/np.sqrt(G*(Msun)/r**3)
n = 2*np.pi/T
year = 365.25*24.*60.*60.
Rhill = r*((m1+m2)/Msun/3.)**(1./3.)
mu1 = G*(m1+m2)
mu2 = G*(m1+mimp)
mu3 = G*(m2+mimp)
a1 = mu1*r1/(2*mu1 - r1*v1**2)
a2 = mu2*r2/(2*mu2 - r2*v2**2)
a3 = mu3*r3/(2*mu3 - r3*v3**2)
energy1 = -mu1/2/a1
energy2 = -mu2/2/a2
energy3 = -mu3/2/a3

boundps = np.logical_and(energy1 < 0, r1 < Rhill)
boundpimp = np.logical_and(energy2 < 0, r2 < Rhill)
boundsimp = np.logical_and(energy3 < 0, r3 < Rhill)

ref = np.zeros((len(data),3))
OmegaK = np.sqrt(G*(Msun+m1+m2)/r**3)
angles = -OmegaK*times
ref[:,0] = -r + r*np.cos(angles)
ref[:,1] = 0 - r*np.sin(angles)

sp = (s-p)/Rhill
ip = (imp-p)/Rhill
cosspx, cosspy = np.cos(angles)*sp[:,0], np.cos(angles)*sp[:,1]
sinspx, sinspy = np.sin(angles)*sp[:,0], np.sin(angles)*sp[:,1]

xdot = vs[:,0] - vp[:,0]
ydot = vs[:,1] - vp[:,1]
cosspxdot, cosspydot = np.cos(angles)*xdot, np.cos(angles)*ydot
sinspxdot, sinspydot = np.sin(angles)*xdot, np.sin(angles)*ydot

x, y = cosspx-sinspy+r, sinspx+cosspy
vx, vy = cosspxdot-sinspydot, sinspxdot+cosspydot

Cj = n**2*(x**2 + y**2) + 2*(mu1/r1 + mu2/r2) - vx**2 - vy**2
# %%
plt.figure(figsize=(15,8))
plt.title(f"Impactor radius={simp} km -- b={b} hill radii")
plt.plot(times/year, boundps, label="Primary-Secondary", lw=3, alpha=0.7)
plt.plot(times/year, boundpimp, label="Primary-Impactor", lw=3, alpha=0.7)
plt.plot(times/year, boundsimp, label="Secondary-Impactor", lw=3, alpha=0.7)
plt.axhline(y=0, ls="--", color="black", lw=1.5)
plt.xlabel("Time (years)")
plt.ylabel("Energy (J/kg)")
plt.xlim(0, np.amax(times)/year)
# plt.ylim(-0.02, 0.02)
plt.grid('both')
plt.legend()
# plt.savefig(f"energy_{str(sim.integrator)}", bbox_inches='tight')
# %%
plt.figure(figsize=(15,8))
plt.title(f"Impactor radius={simp} km -- b={b} hill radii")
plt.plot(times/year, energy1, label="Primary-Secondary", lw=1.5)
plt.plot(times/year, energy2, label="Primary-Impactor", lw=1.5)
plt.plot(times/year, energy3, label="Secondary-Impactor", lw=1.5)
plt.axhline(y=0, ls="--", color="black", lw=1.5)
plt.xlabel("Time (years)")
plt.ylabel("Energy (J/kg)")
plt.xlim(0, np.amax(times)/year)
# plt.ylim(-0.02, 0.02)
plt.grid('both')
plt.legend()
# plt.savefig(f"energy_{str(sim.integrator)}", bbox_inches='tight')
# %%
plt.figure(figsize=(15,8))
plt.title(f"Impactor radius={simp} km -- b={b} hill radii")
plt.plot(times/year, r1/Rhill, label="Primary-Secondary", lw=1.5)
plt.plot(times/year, r2/Rhill, label="Primary-Impactor", lw=1.5)
plt.plot(times/year, r3/Rhill, label="Secondary-Impactor", lw=1.5)
plt.xlabel("Time (years)")
plt.ylabel("Distance (Hill Radii)")
plt.xlim(0, np.amax(times)/year)
plt.ylim(0)
plt.grid('both')
plt.legend()
# plt.savefig(f"distance_{str(sim.integrator)}", bbox_inches='tight')
# %%
lim = 10
fig, axes = plt.subplots(1, figsize=(9, 9))
axes.set_xlabel("$x/R_\mathrm{h}$")
axes.set_ylabel("$y/R_\mathrm{h}$")
axes.set_ylim(-lim,lim)
axes.set_xlim(-lim,lim)
primaryline, = axes.plot([], [], label="primary", c="tab:orange", lw=1.5)
secondaryline, = axes.plot([], [], label="secondary", c="tab:blue", lw=1.5)
impactorline, = axes.plot([], [], label="impactor", c="tab:green", lw=1.5)
primarydot, = axes.plot([], [], marker="o", ms=8, c="tab:orange")
secondarydot, = axes.plot([], [], marker="o", ms=8, c="tab:blue")
impactordot, = axes.plot([], [], marker="o", ms=8, c="tab:green")
text = axes.text(-lim+(lim/10), lim-(lim/10), '', fontsize=15)
axes.grid()
axes.legend()

pref = (p-ref)/Rhill
sref = (s-ref)/Rhill
impref = (imp-ref)/Rhill
cospx, cospy = np.cos(angles)*pref[:,0], np.cos(angles)*pref[:,1]
cossx, cossy = np.cos(angles)*sref[:,0], np.cos(angles)*sref[:,1]
cosix, cosiy = np.cos(angles)*impref[:,0], np.cos(angles)*impref[:,1]
sinpx, sinpy = np.sin(angles)*pref[:,0], np.sin(angles)*pref[:,1]
sinsx, sinsy = np.sin(angles)*sref[:,0], np.sin(angles)*sref[:,1]
sinix, siniy = np.sin(angles)*impref[:,0], np.sin(angles)*impref[:,1]

def animate(i):
    primaryline.set_data(cospx[0:i]-sinpy[0:i], sinpx[0:i]+cospy[0:i])
    secondaryline.set_data(cossx[0:i]-sinsy[0:i], sinsx[0:i]+cossy[0:i])
    impactorline.set_data(cosix[0:i]-siniy[0:i], sinix[0:i]+cosiy[0:i])
    primarydot.set_data(cospx[i]-sinpy[i], sinpx[i]+cospy[i])
    secondarydot.set_data(cossx[i]-sinsy[i], sinsx[i]+cossy[i])
    impactordot.set_data(cosix[i]-siniy[i], sinix[i]+cosiy[i])
    text.set_text('{} Years'.format(np.round(times[i]/(year), 1)))
    return primarydot, secondarydot, impactordot, primaryline, secondaryline, impactorline, text

anim = animation.FuncAnimation(fig, animate, blit=True, frames=len(data), interval=1)
# anim.save('test.mp4')
# %%
lim = 2
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

# sp = (s-p)/Rhill
# ip = (imp-p)/Rhill

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
    
    # secondaryline.set_data(cosspx[0:i]-sinspy[0:i], sinspx[0:i]+cosspy[0:i])
    # impactorline.set_data(cosipx[0:i]-sinipy[0:i], sinipx[0:i]+cosipy[0:i])
    # # primaryline.set_3d_properties(zprimary[0:i])
    # secondaryline.set_3d_properties(sp[0:i,2])
    # impactorline.set_3d_properties(ip[0:i,2])
    # secondary.set_data(cosspx[i]-sinspy[i], sinspx[i]+cosspy[i])
    # impactor.set_data(cosipx[i]-sinipy[i], sinipx[i]+cosipy[i])
    # primary.set_3d_properties(0)
    # secondary.set_3d_properties(sp[i,2])
    # impactor.set_3d_properties(ip[i,2])
    text.set_text('{} Years'.format(np.round(times[i]/(year), 1)))
    return primarydot, secondarydot, impactordot, primaryline, secondaryline, impactorline, text

anim = animation.FuncAnimation(fig, animate, blit=True, frames=len(data), interval=1)
# %%
anim.save('3D.mp4')