# %%
import glob, os, csv, rebound
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from timeit import default_timer as timed
from matplotlib import animation
from matplotlib.animation import FuncAnimation, FFMpegWriter

G = 6.67428e-11                             # gravitational constanct in SI units
au = 1.496e11                               # astronomical unit    
Msun = 1.9891e30                            # mass of sun
year = 365.25*24.*60.*60.                   # number of seconds in a year
s1, s2 = 130e3, 100e3                         # radius of primary and of secondary
dens1, dens2, densimp = 1000., 1000., 1000. # density of primary, secondary, and impactor 
m1 = 4./3.*np.pi*dens1*s1**3                # mass of primary calculated from density and radius
m2 = 4./3.*np.pi*dens2*s2**3                # mass of secondary calculated from density and radius
rsun = 44.*au                                  # distance of centre of mass of binary from the sun 
OmegaK = np.sqrt(G*Msun/rsun**3)       # keplerian frequency at this distance
Rhill1 = rsun*(m1/Msun/3.)**(1./3.)        # Hill radius of primary
rbin = 0.3*Rhill1                            # separation of binary
# rbin = 135000000
vorb = np.sqrt(G*(m1+m2)/rbin)              # orbital speed of primary and secondary around each other
vshear = -1.5*OmegaK*rbin                   # calculates the change in velocity required to keep a body in a circular orbit
Pbin = 2.*np.pi/np.sqrt(G*(m1+m2)/rbin**3)  # orbital period of primary and secondary around each other
T = 2.*np.pi/np.sqrt(G*Msun/rsun**3)         # orbital period of binary around the sun
n = 2*np.pi/T                               # mean motion of binary around the sun

simp = 300e3 # impactor radius
b = 1*Rhill1 # impact parameter
        
y0 = Rhill1*simp/1e3                  # initial y distance of impactor from binary - larger for larger impactors
y0 = Rhill1
mimp = 4./3.*np.pi*densimp*simp**3   # mass of impactor

xb1 = -m2/(m1+m2)*rbin                  # slightly adjust initial x position of primary to keep centre of mass of binary at r
xb2 = m1/(m1+m2)*rbin                   # slightly adjust initial x position of secondary to keep centre of mass of binary at r

vshear1 = -1.5*OmegaK*xb1               # keplerian shear of primary
vshear2 = -1.5*OmegaK*xb2               # keplerian shear of secondary

vK1 = np.sqrt(G*(Msun+m1)/(rsun+xb1))      # orbital speed of primary around sun
vK2 = np.sqrt(G*(Msun+m2)/(rsun+xb2))      # inital orbital speed of secondary around sun

binaryi = np.deg2rad(0)     # inclination of binary
vorb1 = -m2/(m1+m2)*vorb                # orbital speed of primary around secondary - adjusted to account for offset from COM
vorb2 = m1/(m1+m2)*vorb                 # orbital speed of secondary around primary - adjusted to account for offset from COM
sinbin = np.sin(binaryi)                # sin of inclination of binary
cosbin = np.cos(binaryi)                # cos of inclination of binary

primx = xb1*np.sin(np.pi/2-binaryi)     # x position of primary - accounts for inclination
primz = xb1*np.sin(binaryi)             # z position of primary - accounts for inclination
primvy = vK1+vorb1*cosbin               # y velocity of primary - vy is keplerian velocity plus vorb
primvz = -vorb1*sinbin                  # z velocity of primary - added if i > 0

secx = xb2*np.sin(np.pi/2-binaryi)     # x position of secondary - accounts for inclination
secz = xb2*np.sin(binaryi)             # z position of secondary - accounts for inclination
secvy = vK2+vorb2*cosbin               # y velocity of secondary - vy is keplerian velocity plus vorb
secvz = -vorb2*sinbin                  # z velocity of secondary - added if i > 0

impi = np.deg2rad(0)        # inclination of impactor
vorbi = np.sqrt(G*Msun/(rsun+b))        # orbital speed of impactor around sun
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
# %%
sim = rebound.Simulation()              # initialize rebound simulation
sim.G = G                               # set G which sets units of integrator - SI in this case
sim.dt = 1e-4*Pbin                      # set initial timestep of integrator - IAS15 is adaptive so this will change
sim.softening = 0.1*s1                  # softening parameter which modifies potential of each particle to prevent divergences
# sim.collision = 'direct'
# sim.collision_resolve = 'merge'
sim.add(m=Msun, x=-rsun, hash="sun")
sim.add(m=m1, r=s1, x=primx, z=primz, vy=primvy, vz=primvz, hash="primary")
sim.add(m=m2, r=s2, x=secx, z=secz, vy=secvy, vz=secvz, hash="secondary")
sim.add(m=mimp, r=simp, x=impx, y=impy, z=impz, vx=impvx, vy=impvy, hash="impactor")

Noutputs = 1000             # number of outputs
p, s, imp, sun = np.zeros((Noutputs, 3)), np.zeros((Noutputs, 3)), np.zeros((Noutputs, 3)), np.zeros((Noutputs, 3)) # position
vp, vs, vimp, vsun = np.zeros((Noutputs, 3)), np.zeros((Noutputs, 3)), np.zeros((Noutputs, 3)), np.zeros((Noutputs, 3)) # velocity
totaltime = T*simp/10e3*(1/b*Rhill1)*3. # total time of simulation - adjusted for different impactor sizes and distances
totaltime = T/10
times = np.linspace(0.,totaltime, Noutputs) # create times for integrations
ps = sim.particles                      # create variable containing particles in simulation
timer = timed() # start timer to time simulations
for i, time in enumerate(times):
    sim.integrate(time)
    p[i] = [ps["primary"].x, ps["primary"].y, ps["primary"].z]
    s[i] = [ps["secondary"].x, ps["secondary"].y, ps["secondary"].z]
    imp[i] = [ps["impactor"].x, ps["impactor"].y, ps["impactor"].z]
    sun[i] = [ps["sun"].x, ps["sun"].y, ps["sun"].z]
    vp[i] = [ps["primary"].vx, ps["primary"].vy, ps["primary"].vz]
    vs[i] = [ps["secondary"].vx, ps["secondary"].vy, ps["secondary"].vz]
    vimp[i] = [ps["impactor"].vx, ps["impactor"].vy, ps["impactor"].vz]
    vsun[i] = [ps["sun"].vx, ps["sun"].vy, ps["sun"].vz]
            
print(timed()-timer) # finish timer

R, V, mu, h = np.zeros((Noutputs,3)), np.zeros((Noutputs,3)), np.zeros((Noutputs,3)), np.zeros((Noutputs,3))

R[:,0] = np.linalg.norm(p-s, axis=1)                # distance between primary and secondary
R[:,1] = np.linalg.norm(p-imp, axis=1)              # distance between primary and impactor
R[:,2] = np.linalg.norm(s-imp, axis=1)              # distance between secondary and impactor

V[:,0] = np.linalg.norm(vp-vs, axis=1)              # relative velocity between primary and secondary
V[:,1] = np.linalg.norm(vp-vimp, axis=1)            # relative velocity between primary and impactor
V[:,2] = np.linalg.norm(vs-vimp, axis=1)            # relative velocity between secondary and impactor

mu[:,0] = G*(m1+m2)                                 # G times combined mass of primary and secondary
mu[:,1] = G*(m1+mimp)                               # G times combined mass of primary and impactor
mu[:,2] = G*(m2+mimp)                               # G times combined mass of secondary and impactor

h[:,0] = np.cross(p-s,vp-vs)[:,2]                       # angular momentum
h[:,1] = np.cross(p-imp,vp-vimp)[:,2]
h[:,2] = np.cross(s-imp,vs-vimp)[:,2]

semimajoraxis = mu*R/(2*mu-R*V**2)                           # semi-major axis between each pair of bodies
energy = -mu/2/semimajoraxis                                    # total energy between each pair of bodies

collisions = R < np.array([s1+s2, s1+simp, s2+simp])
collision_speed = V[collisions]

Rhill = np.array([Rhill1, rsun*(m2/Msun/3.)**(1./3.), rsun*(mimp/Msun/3.)**(1./3.)])
bound = np.logical_and(energy < 0, R < Rhill)       # bodies are bound if their energy is less than zero and they are closer together than the Hill radius

angles = -OmegaK*times

sp = (s-p)/Rhill1
ip = (imp-p)/Rhill1
cosspx, cosspy = np.cos(angles)*sp[:,0], np.cos(angles)*sp[:,1]
sinspx, sinspy = np.sin(angles)*sp[:,0], np.sin(angles)*sp[:,1]

xdot = vs[:,0] - vp[:,0]
ydot = vs[:,1] - vp[:,1]
cosspxdot, cosspydot = np.cos(angles)*xdot, np.cos(angles)*ydot
sinspxdot, sinspydot = np.sin(angles)*xdot, np.sin(angles)*ydot

x, y = cosspx-sinspy+rsun, sinspx+cosspy
vx, vy = cosspxdot-sinspydot, sinspxdot+cosspydot

Cj = n**2*(x**2+y**2) + 2*(mu[:,0]/R[:,0] + mu[:,1]/R[:,1]) - vx**2 - vy**2 # jacobian constant
# %%
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
# axes.grid()
axes.legend()

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
primaryhill = plt.Circle((0,0), Rhillprim, fc="none", ec="tab:orange")
secondaryhill = plt.Circle((0,0), Rhillsec, fc="none", ec="tab:blue")
impactorhill = plt.Circle((0,0), Rhillimp, fc="none", ec="tab:green")

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

anim = animation.FuncAnimation(fig, animate, frames=Noutputs, interval=1, blit=True)
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
