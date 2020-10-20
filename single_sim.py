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
year = 365.25*24.*60.*60.                   # number of seconds in a year
s1, s2 = 100e3, 100e3                         # radius of primary and of secondary
dens1, dens2, densimp = 1000., 1000., 0. # density of primary, secondary, and impactor 
m1 = 4./3.*np.pi*dens1*s1**3                # mass of primary calculated from density and radius
m2 = 4./3.*np.pi*dens2*s2**3                # mass of secondary calculated from density and radius
rsun = 44.*au                                  # distance of centre of mass of binary from the sun 
omegak = np.sqrt(g*msun/rsun**3)       # keplerian frequency at this distance
rhill1 = rsun*(m1/msun/3.)**(1./3.)        # Hill radius of primary
rbin = 0.3*rhill1                            # separation of binary
vorb = np.sqrt(g*(m1+m2)/rbin)              # orbital speed of primary and secondary around each other
# vshear = -1.5*omegak*rbin                   # calculates the change in velocity required to keep a body in a circular orbit
pbin = 2.*np.pi/np.sqrt(g*(m1+m2)/rbin**3)  # orbital period of primary and secondary around each other
t = 2.*np.pi/np.sqrt(g*msun/rsun**3)         # orbital period of binary around the sun
n = 2*np.pi/t                               # mean motion of binary around the sun
vk = np.sqrt(g*msun/rsun)      # orbital speed of primary around sun
simp = 100e3 # impactor radius
b = 15*rhill1 # impact parameter
        
y0 = rhill1*simp/1e3                  # initial y distance of impactor from binary - larger for larger impactors
y0 = 0
mimp = 4./3.*np.pi*densimp*simp**3   # mass of impactor

# vshear1 = -1.5*omegak*xb1               # keplerian shear of primary
# vshear2 = -1.5*omegak*xb2               # keplerian shear of secondary


binaryi = np.deg2rad(0)                 # inclination of binary
sinbin = np.sin(binaryi)                # sin of inclination of binary
cosbin = np.cos(binaryi)                # cos of inclination of binary

xb1 = -m2/(m1+m2)*rbin                  # slightly adjust initial x position of primary to keep centre of mass of binary at r
vorb1 = -m2/(m1+m2)*vorb                # orbital speed of primary around secondary - adjusted to account for offset from COM
primx = xb1*np.cos(binaryi)     # x position of primary - accounts for inclination
primz = xb1*np.sin(binaryi)             # z position of primary - accounts for inclination
primvy = vk+vorb1*cosbin               # y velocity of primary - vy is keplerian velocity plus vorb
primvz = -vorb1*sinbin                  # z velocity of primary - added if i > 0

xb2 = m1/(m1+m2)*rbin                   # slightly adjust initial x position of secondary to keep centre of mass of binary at r
vk2 = np.sqrt(g*(msun)/(rsun))      # inital orbital speed of secondary around sun
vorb2 = m1/(m1+m2)*vorb                 # orbital speed of secondary around primary - adjusted to account for offset from COM
secx = xb2*np.cos(binaryi)     # x position of secondary - accounts for inclination
secz = xb2*np.sin(binaryi)             # z position of secondary - accounts for inclination
secvy = vk+vorb2*cosbin               # y velocity of secondary - vy is keplerian velocity plus vorb
secvz = -vorb2*sinbin                  # z velocity of secondary - added if i > 0

impi = np.deg2rad(0)        # inclination of impactor
vorbi = np.sqrt(g*msun/(rsun+b))        # orbital speed of impactor around sun
theta0 = y0/(rsun+b)                    # angle between impactor and line between binary COM and sun
stheta0 = np.sin(theta0)                # sin of theta - needed for position of impactor
ctheta0 = np.cos(theta0)                # cos of theta - needed for position of impactor
impx = (rsun+b)*ctheta0-rsun*np.cos(impi)  # x position of impactor
impy = (rsun+b)*stheta0*np.cos(impi)    # y position of impactor
impz = (rsun+b)*np.sin(impi)            # z position of impactor
impvx = -vorbi*stheta0                  # x velocity of impactor
impvy = vorbi*ctheta0                   # y velocity of impactor

# Cd = 2.
# rho_g = 1e-20
# drag1 = 0.5*Cd*np.pi*s1**2*rho_g
# drag2 = 0.5*Cd*np.pi*s2**2*rho_g
# drag3 = 0.5*Cd*np.pi*simp**2*rho_g

# def dragForce(reb_sim):
#     ps["primary"].ax -= (ps["primary"].vx)**2*drag1
#     ps["primary"].ay -= (ps["primary"].vy)**2*drag1
#     ps["primary"].az -= (ps["primary"].vz)**2*drag1
#     ps["secondary"].ax -= (ps["secondary"].vx)**2*drag2
#     ps["secondary"].ay -= (ps["secondary"].vy)**2*drag2
#     ps["secondary"].az -= (ps["secondary"].vz)**2*drag2
#     ps["impactor"].ax -= (ps["impactor"].vx)**2*drag3
#     ps["impactor"].ay -= (ps["impactor"].vy)**2*drag3
#     ps["impactor"].az -= (ps["impactor"].vz)**2*drag3
    
# sim.additional_forces = dragForce
# sim.force_is_velocity_dependent = 1

def setupSimulation():
    sim = rebound.Simulation()              # initialize rebound simulation
    sim.G = g                               # set G which sets units of integrator - SI in this case
    sim.dt = 1e-4*pbin                      # set initial timestep of integrator - IAS15 is adaptive so this will change
    sim.softening = 0.1*s1                  # softening parameter which modifies potential of each particle to prevent divergences
    sim.collision = 'direct'
    sim.add(m=msun, x=-rsun, hash="sun")
    sim.add(m=m1, r=s1, x=primx, vy=primvy, hash="primary")
    sim.add(m=m2, r=s2, x=secx, vy=secvy, hash="secondary")
    sim.add(m=mimp, r=simp, x=impx, y=impy, z=impz, vx=impvx, vy=impvy, hash="impactor")
    return sim

sim = setupSimulation()

noutputs = 1000             # number of outputs
p, s, imp = np.zeros((noutputs, 3)), np.zeros((noutputs, 3)), np.zeros((noutputs, 3)) # position
vp, vs, vimp = np.zeros((noutputs, 3)), np.zeros((noutputs, 3)), np.zeros((noutputs, 3)) # velocity
# totaltime = t*simp/10e3*(1/b*rhill1)*3. # total time of simulation - adjusted for different impactor sizes and distances
totaltime = t
times = np.linspace(0.,totaltime, noutputs) # create times for integrations
ps = sim.particles                      # create variable containing particles in simulation

all_ps = [p.hash.value for j, p in enumerate(ps)]

timer = timed() # start timer to time simulations
try:
    for k, time in enumerate(times):
        sim.integrate(time)
        p[k] = [ps["primary"].x, ps["primary"].y, ps["primary"].z]
        s[k] = [ps["secondary"].x, ps["secondary"].y, ps["secondary"].z]
        imp[k] = [ps["impactor"].x, ps["impactor"].y, ps["impactor"].z]
        vp[k] = [ps["primary"].vx, ps["primary"].vy, ps["primary"].vz]
        vs[k] = [ps["secondary"].vx, ps["secondary"].vy, ps["secondary"].vz]
        vimp[k] = [ps["impactor"].vx, ps["impactor"].vy, ps["impactor"].vz]
except rebound.Collision:
    collided = []
    for item in sim.particles:
        if item.lastcollision == sim.t:
            collided.append([sim.t, item.index, item.r, item.m, item.x, item.y, item.z, item.vx, item.vy, item.vz])
    collided = np.array(collided) 
    
    sim.collision_resolve = 'merge'

    for k, time in enumerate(times):
        sim.integrate(time)
        existing_ps = [p.hash.value for j, p in enumerate(ps)]
        if all_ps[1] in existing_ps:
            p[k] = [ps["primary"].x, ps["primary"].y, ps["primary"].z]
            vp[k] = [ps["primary"].vx, ps["primary"].vy, ps["primary"].vz]
        if all_ps[2] in existing_ps:
            s[k] = [ps["secondary"].x, ps["secondary"].y, ps["secondary"].z]
            vs[k] = [ps["secondary"].vx, ps["secondary"].vy, ps["secondary"].vz]
        if all_ps[3] in existing_ps:
            imp[k] = [ps["impactor"].x, ps["impactor"].y, ps["impactor"].z]
            vimp[k] = [ps["impactor"].vx, ps["impactor"].vy, ps["impactor"].vz]
print(timed()-timer) # finish timer

dr, dv, mu, h = np.zeros((noutputs,3)), np.zeros((noutputs,3)), np.zeros((noutputs,3)), np.zeros((noutputs,3))
dr[:,0] = np.linalg.norm(p-s, axis=1)                # distance between primary and secondary
dr[:,1] = np.linalg.norm(p-imp, axis=1)              # distance between primary and impactor
dr[:,2] = np.linalg.norm(s-imp, axis=1)              # distance between secondary and impactor
dv[:,0] = np.linalg.norm(vp-vs, axis=1)              # relative velocity between primary and secondary
dv[:,1] = np.linalg.norm(vp-vimp, axis=1)            # relative velocity between primary and impactor
dv[:,2] = np.linalg.norm(vs-vimp, axis=1)            # relative velocity between secondary and impactor
mu[:,0] = g*(m1+m2)                                 # G times combined mass of primary and secondary
mu[:,1] = g*(m1+mimp)                               # G times combined mass of primary and impactor
mu[:,2] = g*(m2+mimp)                               # G times combined mass of secondary and impactor
h[:,0] = np.cross(p-s,vp-vs)[:,2]                       # angular momentum
h[:,1] = np.cross(p-imp,vp-vimp)[:,2]
h[:,2] = np.cross(s-imp,vs-vimp)[:,2]

semimajoraxis = mu*dr/(2*mu-dr*dv**2)                           # semi-major axis between each pair of bodies
energy = -mu/2/semimajoraxis                                    # total energy between each pair of bodies 
ecc = np.sqrt(1+(2*energy*h**2/mu**2))
angles = -omegak*times

rhill = np.array([rhill1, rsun*(m2/msun/3.)**(1./3.), rsun*(mimp/msun/3.)**(1./3.)])
rhill_largest = np.array([np.amax([rhill[0], rhill[1]]), np.amax([rhill[0], rhill[2]]), np.amax([rhill[1], rhill[2]])])
bound = np.logical_and(np.logical_and(energy < 0, np.isfinite(energy)), dr < rhill_largest)       # bodies are bound if their energy is less than zero and they are closer together than the Hill radius

sp = (s-p)/rhill1
ip = (imp-p)/rhill1
cosspx, cosspy = np.cos(angles)*sp[:,0], np.cos(angles)*sp[:,1]
sinspx, sinspy = np.sin(angles)*sp[:,0], np.sin(angles)*sp[:,1]

xdot = vs[:,0] - vp[:,0]
ydot = vs[:,1] - vp[:,1]
cosspxdot, cosspydot = np.cos(angles)*xdot, np.cos(angles)*ydot
sinspxdot, sinspydot = np.sin(angles)*xdot, np.sin(angles)*ydot

x, y = cosspx-sinspy+rsun, sinspx+cosspy
vx, vy = cosspxdot-sinspydot, sinspxdot+cosspydot

# plt.plot(ecc[:,0])

# cj = n**2*(x**2+y**2) + 2*(mu[:,0]/dr[:,0] + mu[:,1]/dr[:,1]) - vx**2 - vy**2 # jacobian constant

lim = 4
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
axes.grid()
axes.legend()

ref = np.zeros((noutputs,3))
ref[:,0] = -rsun + rsun*np.cos(angles)
ref[:,1] = 0 - rsun*np.sin(angles)

pref = (p-ref)/rhill1
sref = (s-ref)/rhill1
impref = (imp-ref)/rhill1
cospx, cospy = np.cos(angles)*pref[:,0], np.cos(angles)*pref[:,1]
cossx, cossy = np.cos(angles)*sref[:,0], np.cos(angles)*sref[:,1]
cosix, cosiy = np.cos(angles)*impref[:,0], np.cos(angles)*impref[:,1]
sinpx, sinpy = np.sin(angles)*pref[:,0], np.sin(angles)*pref[:,1]
sinsx, sinsy = np.sin(angles)*sref[:,0], np.sin(angles)*sref[:,1]
sinix, siniy = np.sin(angles)*impref[:,0], np.sin(angles)*impref[:,1]

rhillprim = rsun*(m1/msun/3.)**(1./3.)/rhill1
rhillsec = rsun*(m2/msun/3.)**(1./3.)/rhill1
rhillimp = rsun*(mimp/msun/3.)**(1./3.)/rhill1
primaryhill = plt.Circle((0,0), rhillprim, fc="none", ec="tab:orange")
secondaryhill = plt.Circle((0,0), rhillsec, fc="none", ec="tab:blue")
impactorhill = plt.Circle((0,0), rhillimp, fc="none", ec="tab:green")

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
    return primarydot, secondarydot, impactordot, primaryline, secondaryline, impactorline, primaryhill, secondaryhill, impactorhill, text

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=noutputs, interval=1, blit=True)
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

angles = -omegak*times
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

anim = animation.FuncAnimation(fig, animate, frames=noutputs, interval=1, blit=True)
# anim.save(f'{path}/videos/3D.mp4')
# %%
lim = 1
fig, axes = plt.subplots(1, figsize=(5, 5))
axes.set_xlabel("$x/R_\mathrm{h}$")
axes.set_ylabel("$y/R_\mathrm{h}$")
axes.set_ylim(-lim,lim)
axes.set_xlim(-lim,lim)

ref = np.zeros((Noutputs,3))
ref[:,0] = -rsun + rsun*np.cos(angles)
ref[:,1] = 0 - rsun*np.sin(angles)

pref = (p-ref)/Rhill1
sref = (s-ref)/Rhill1
impref = (imp-ref)/Rhill1
cospx, cospy = np.cos(angles)*pref[:,0], np.cos(angles)*pref[:,1]
cossx, cossy = np.cos(angles)*sref[:,0], np.cos(angles)*sref[:,1]
cosix, cosiy = np.cos(angles)*impref[:,0], np.cos(angles)*impref[:,1]
sinpx, sinpy = np.sin(angles)*pref[:,0], np.sin(angles)*pref[:,1]
sinsx, sinsy = np.sin(angles)*sref[:,0], np.sin(angles)*sref[:,1]
sinix, siniy = np.sin(angles)*impref[:,0], np.sin(angles)*impref[:,1]

Rhillprim = rsun*(m1/Msun/3.)**(1./3.)/Rhill1
Rhillsec = rsun*(m2/Msun/3.)**(1./3.)/Rhill1
Rhillimp = rsun*(mimp/Msun/3.)**(1./3.)/Rhill1
primaryhill = plt.Circle((cospx[-1]-sinpy[-1], sinpx[-1]+cospy[-1]), Rhillprim, fc="none", ec="tab:orange")
axes.add_artist(primaryhill)
secondaryhill = plt.Circle((cossx[-1]-sinsy[-1], sinsx[-1]+cossy[-1]), Rhillsec, fc="none", ec="tab:blue")
axes.add_artist(secondaryhill)
impactorhill = plt.Circle((cosix[-1]-siniy[-1], sinix[-1]+cosiy[-1]), Rhillimp, fc="none", ec="tab:green")
axes.add_artist(impactorhill)

axes.plot(cospx-sinpy, sinpx+cospy, label="primary", c="tab:orange", lw=1.5)
axes.plot(cossx-sinsy, sinsx+cossy, label="secondary", c="tab:blue", lw=1.5)
axes.plot(cosix-siniy, sinix+cosiy, label="impactor", c="tab:green", lw=1.5)

axes.plot(cospx[-1]-sinpy[-1], sinpx[-1]+cospy[-1], c="tab:orange", marker='o')
axes.plot(cossx[-1]-sinsy[-1], sinsx[-1]+cossy[-1], c="tab:blue", marker='o')
axes.plot(cosix[-1]-siniy[-1], sinix[-1]+cosiy[-1], c="tab:green", marker='o')

axes.legend()
# fig.savefig('./result5.pdf', bbox_inches='tight')
# %%
y = Cj
plt.figure(figsize=(15,8))
# plt.title(f"Integrator={sim.integrator}  Imp radius={simp/1e3} km  b={b/Rhill} Rhill")
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
# plt.title(f"Integrator={sim.integrator} -- Impactor radius={simp/1e3} km -- b={b/Rhill} hill radii")
plt.plot(times/year, y[:,0], label="Primary-Secondary", lw=1.5)
plt.plot(times/year, y[:,1], label="Primary-Impactor", lw=1.5)
plt.plot(times/year, y[:,2], label="Secondary-Impactor", lw=1.5)
plt.axhline(y=0, ls="--", color="black", lw=1.5)
plt.xlabel("Time (years)")
plt.ylabel("Energy (J/kg)")
plt.xlim(0, np.amax(times)/year)
# plt.ylim(-0.02, 0.02)
plt.grid('both')
plt.legend()
# plt.savefig(f"energy_{str(sim.integrator)}", bbox_inches='tight')
# %%
y = R
plt.figure(figsize=(15,8))
# plt.title(f"Integrator={sim.integrator} -- Impactor radius={simp/1e3} km -- b={b/Rhill} hill radii")
plt.plot(times/year, y[:,0]/Rhill1, label="Primary-Secondary", lw=1.5)
plt.plot(times/year, y[:,1]/Rhill1, label="Primary-Impactor", lw=1.5)
plt.plot(times/year, y[:,2]/Rhill1, label="Secondary-Impactor", lw=1.5)
plt.xlabel("Time (years)")
plt.ylabel("Distance (Hill Radii)")
plt.xlim(0, np.amax(times)/year)
plt.ylim(0)
plt.grid('both')
plt.legend()
# plt.savefig(f"distance_{str(sim.integrator)}", bbox_inches='tight')
