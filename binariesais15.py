# %%
import rebound
import reboundx
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation
from scipy.spatial.distance import pdist
from timeit import default_timer as timed

G = 6.67428e-11
au = 1.496e11
Msun = 1.9891e30
year = 365.25*24.*60.*60.
s1, s2 = 10e3, 10e3
dens1, dens2, densimp = 1000., 1000., 1000.
m1, m2  = 4./3.*np.pi*dens1*s1**3,  4./3.*np.pi*dens2*s2**3
r = 44.*au
OmegaK = np.sqrt(G*(Msun+m1+m2)/r**3)
Rhill = r*((m1+m2)/Msun/3.)**(1./3.)
rbin = 0.9*Rhill
vorb = np.sqrt(G*(m1+m2)/rbin)
vshear = -1.5*OmegaK*rbin
Pbin = 2.*np.pi/np.sqrt(G*(m1+m2)/rbin**3)
T = 2.*np.pi/np.sqrt(G*(Msun)/r**3)
n = 2*np.pi/T
y0 = 10*Rhill
mu1 = G*Msun
mu2 = G*m1
# %%
simp = 10e3
mimp = 4./3.*np.pi*densimp*simp**3

B = 4.0*Rhill

binaryi = np.deg2rad(0)
impi = np.deg2rad(0)

sim = rebound.Simulation()
rebx = reboundx.Extras(sim)
# sim.testparticle_type = 0
sim.G = G
sim.dt = 1e-4*Pbin 
sim.softening = 0.1*s1
sim.integrator = "ias15"
sim.gravity    = "basic"
sim.collision  = "none"

xb1 = -m2/(m1+m2)*rbin
xb2 = m1/(m1+m2)*rbin

vcom = r*OmegaK
vshear1 = -1.5*OmegaK*xb1
vshear2 = -1.5*OmegaK*xb2

vK1 = np.sqrt(G*(Msun+m1)/(r+xb1))
vK2 = np.sqrt(G*(Msun+m2)/(r+xb2))

vorb1 = -m2/(m1+m2)*vorb
vorb2 = m1/(m1+m2)*vorb
sinbin = np.sin(binaryi)
cosbin = np.cos(binaryi)

sim.add(m=Msun, x=-r, hash="sun")
sim.add(m=m1, r=s1, x=xb1*np.sin(np.pi/2-binaryi), z=xb1*np.sin(binaryi), vy=vK1+vorb1*cosbin, vz=-vorb1*sinbin, hash="primary")
# sim.N_active = sim.N
sim.add(m=m2, r=s2, x=xb2*np.sin(np.pi/2-binaryi), z=xb2*np.sin(binaryi), vy=vK2+vorb2*cosbin, vz=-vorb2*sinbin, hash="secondary")

vorbi = np.sqrt(G*Msun/(r+B))
theta0 = y0/(r+B)
stheta0 = np.sin(theta0)
ctheta0 = np.cos(theta0)

sim.add(m=mimp, r=simp, x=(r+B)*ctheta0-r*np.cos(impi), y=(r+B)*stheta0*np.cos(impi), z=(r+B)*np.sin(impi), vx=-vorbi*stheta0, vy=vorbi*ctheta0, hash="impactor")


Noutputs = 1000
totaltime = T
p, s, imp, sun = np.zeros((Noutputs, 3)), np.zeros((Noutputs, 3)), np.zeros((Noutputs, 3)), np.zeros((Noutputs, 3))
vp, vs, vimp, vsun = np.zeros((Noutputs, 3)), np.zeros((Noutputs, 3)), np.zeros((Noutputs, 3)), np.zeros((Noutputs, 3))
times = np.linspace(0.,totaltime, Noutputs)
ps = sim.particles
distances = np.zeros((Noutputs, 6))
energy = np.zeros((Noutputs, 3))
Cj = np.zeros((Noutputs))

for i, time in enumerate(times):
    sim.integrate(time)    
    p[i] = [ps["primary"].x, ps["primary"].y, ps["primary"].z]
    s[i] = [ps["secondary"].x, ps["secondary"].y, ps["secondary"].z]
    imp[i] = [ps["impactor"].x, ps["impactor"].y, ps["impactor"].z]
    sun[i] = [ps["sun"].x, ps["sun"].y, ps["sun"].z]    
    particles = [p[i], s[i], imp[i], sun[i]]
    distances[i] = pdist(particles)    
    vp[i] = [ps["primary"].vx, ps["primary"].vy, ps["primary"].vz]
    vs[i] = [ps["secondary"].vx, ps["secondary"].vy, ps["secondary"].vz]
    vimp[i] = [ps["impactor"].vx, ps["impactor"].vy, ps["impactor"].vz]
    vsun[i] = [ps["sun"].vx, ps["sun"].vy, ps["sun"].vz]    
    energy[i,0] = -G*(m1+m2)/2/ps["primary"].calculate_orbit(primary=ps["secondary"]).a
    energy[i,1] = -G*(m1+mimp)/2/ps["primary"].calculate_orbit(primary=ps["impactor"]).a
    energy[i,2] = -G*(m2+mimp)/2/ps["secondary"].calculate_orbit(primary=ps["impactor"]).a
    
sp = s-p
ip = imp-p
angles = -OmegaK*times

cosspx, cosspy = np.cos(angles)*sp[:,0], np.cos(angles)*sp[:,1]
sinspx, sinspy = np.sin(angles)*sp[:,0], np.sin(angles)*sp[:,1]

xdot = vs[:,0] - vp[:,0]
ydot = vs[:,1] - vp[:,1]
cosspxdot, cosspydot = np.cos(angles)*xdot, np.cos(angles)*ydot
sinspxdot, sinspydot = np.sin(angles)*xdot, np.sin(angles)*ydot

x, y = cosspx-sinspy+r, sinspx+cosspy
vx, vy = cosspxdot-sinspydot, sinspxdot+cosspydot
mu1 = G*Msun
mu2 = G*m1
r1 = distances[:,4]
r2 = distances[:,0]

Cj = n**2*(x**2 + y**2) + 2*(mu1/r1 + mu2/r2) - vx**2 - vy**2
# %%
y = Cj
plt.figure(figsize=(15,8))
plt.title(f"Integrator={sim.integrator}  Imp radius={simp/1e3} km  b={B/Rhill} Rhill")
plt.plot(times/year, y, label="Jacobi integral", lw=1.5)
plt.axhline(y=0, ls="--", color="black", lw=1.5)
plt.xlabel("Time (years)")
plt.ylabel("Energy (J/kg)")
plt.xlim(0, np.amax(times)/year)
# plt.ylim(-0.02, 0.02)
plt.grid('both')
plt.legend()
# plt.savefig(f"energy_{str(sim.integrator)}", bbox_inches='tight')
# %%
y = energy
plt.figure(figsize=(15,8))
plt.title(f"Integrator={sim.integrator} -- Impactor radius={simp/1e3} km -- b={B/Rhill} hill radii")
plt.plot(times/year, y[:,0], label="Primary-Secondary", lw=1.5)
plt.plot(times/year, y[:,1], label="Primary-Impactor", lw=1.5)
plt.plot(times/year, y[:,2], label="Secondary-Impactor", lw=1.5)
plt.axhline(y=0, ls="--", color="black", lw=1.5)
plt.xlabel("Time (years)")
plt.ylabel("Energy (J/kg)")
plt.xlim(0, np.amax(times)/year)
plt.ylim(-0.02, 0.02)
plt.grid('both')
plt.legend()
# plt.savefig(f"energy_{str(sim.integrator)}", bbox_inches='tight')
# %%
y = distances
plt.figure(figsize=(15,8))
plt.title(f"Integrator={sim.integrator} -- Impactor radius={simp/1e3} km -- b={B/Rhill} hill radii")
plt.plot(times/year, y[:,0]/Rhill, label="Primary-Secondary", lw=1.5)
plt.plot(times/year, y[:,1]/Rhill, label="Primary-Impactor", lw=1.5)
plt.plot(times/year, y[:,3]/Rhill, label="Secondary-Impactor", lw=1.5)
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

sp = (s-p)/Rhill
ip = (imp-p)/Rhill

angles = -OmegaK*times
cosspx, cosspy = np.cos(angles)*sp[:,0], np.cos(angles)*sp[:,1]
cosipx, cosipy = np.cos(angles)*ip[:,0], np.cos(angles)*ip[:,1]
sinspx, sinspy = np.sin(angles)*sp[:,0], np.sin(angles)*sp[:,1]
sinipx, sinipy = np.sin(angles)*ip[:,0], np.sin(angles)*ip[:,1]

def animate(i):
    # primaryline.set_data(p[0:i,0], p[0:i,1])
    secondaryline.set_data(sp[0:i,0], sp[0:i,1])
    impactorline.set_data(ip[0:i,0], ip[0:i,1])
    secondaryline.set_data(cosspx[0:i]-sinspy[0:i], sinspx[0:i]+cosspy[0:i])
    impactorline.set_data(cosipx[0:i]-sinipy[0:i], sinipx[0:i]+cosipy[0:i])
    primarydot.set_data(0, 0)
    secondarydot.set_data(sp[i,0], sp[i,1])
    impactordot.set_data(ip[i,0], ip[i,1])
    secondarydot.set_data(cosspx[i]-sinspy[i], sinspx[i]+cosspy[i])
    impactordot.set_data(cosipx[i]-sinipy[i], sinipx[i]+cosipy[i])
    text.set_text('{} Years'.format(np.round(times[i]/(year), 1)))
    return primarydot, secondarydot, impactordot, primaryline, secondaryline, impactorline, text

anim = animation.FuncAnimation(fig, animate, frames=Noutputs, blit=True, interval=1)
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
primary, = axes.plot([], [], [], marker="o", ms=8, c="tab:orange")
secondary, = axes.plot([], [], [], marker="o", ms=8, c="tab:blue")
impactor, = axes.plot([], [], [], marker="o", ms=8, c="tab:green")
text = axes.text(-lim+(lim/10), lim-(lim/10), lim-(lim/10), '', fontsize=15)
axes.legend()

sp = (s-p)/Rhill
ip = (imp-p)/Rhill

angles = -OmegaK*times
cosspx, cosspy = np.cos(angles)*sp[:,0], np.cos(angles)*sp[:,1]
cosipx, cosipy = np.cos(angles)*ip[:,0], np.cos(angles)*ip[:,1]
sinspx, sinspy = np.sin(angles)*sp[:,0], np.sin(angles)*sp[:,1]
sinipx, sinipy = np.sin(angles)*ip[:,0], np.sin(angles)*ip[:,1]

def animate(i):
    # primaryline.set_data(xprimary[0:i], yprimary[0:i])
    secondaryline.set_data(sp[0:i,0], sp[0:i,1])
    impactorline.set_data(ip[0:i,0], ip[0:i,1])
    secondaryline.set_data(cosspx[0:i]-sinspy[0:i], sinspx[0:i]+cosspy[0:i])
    impactorline.set_data(cosipx[0:i]-sinipy[0:i], sinipx[0:i]+cosipy[0:i])
    # primaryline.set_3d_properties(zprimary[0:i])
    secondaryline.set_3d_properties(sp[0:i,2])
    impactorline.set_3d_properties(ip[0:i,2])
    primary.set_data(0, 0)
    secondary.set_data(sp[i,0], sp[i,1])
    impactor.set_data(ip[i,0], ip[i,1])
    secondary.set_data(cosspx[i]-sinspy[i], sinspx[i]+cosspy[i])
    impactor.set_data(cosipx[i]-sinipy[i], sinipx[i]+cosipy[i])
    primary.set_3d_properties(0)
    secondary.set_3d_properties(sp[i,2])
    impactor.set_3d_properties(ip[i,2])
    text.set_text('{} Years'.format(np.round(times[i]/(year), 1)))
    return primaryline, secondaryline, impactorline, primary, secondary, impactor, text,

anim = animation.FuncAnimation(fig, animate, frames=Noutputs, interval=10)
# %%
anim.save('3D.mp4')
