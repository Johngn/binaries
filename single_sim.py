# %%
import numpy as np
import rebound
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from powerlaw import rndm
from scipy.stats import powerlaw
from timeit import default_timer as timed
from matplotlib import animation
from matplotlib.animation import FuncAnimation, FFMpegWriter

g = 6.67428e-11                             # gravitational constant in SI units
au = 1.496e11                               # astronomical unit    
msun = 1.9891e30                            # mass of sun
rsun = 44.*au                                  # distance of centre of mass of binary from the sun 
year = 365.25*24.*60.*60.                   # number of seconds in a year
dens = 700.

s1, s2 = 100e3, 100e3                         # radius of primary and of secondary
m1 = 4./3.*np.pi*dens*s1**3                # mass of primary calculated from density and radius
m2 = 4./3.*np.pi*dens*s2**3                # mass of secondary calculated from density and radius
rhill = rsun*((m1+m2)/msun/3.)**(1./3.)        # Hill radius of primary


a = 0.3*rhill                            # separation of binary
e = 0
i = np.deg2rad(0)

pbin = 2.*np.pi/np.sqrt(g*(m1+m2)/a**3)            # orbital period of binary around the sun
t = 2.*np.pi/np.sqrt(g*msun/rsun**3)            # orbital period of binary around the sun
noutputs = 1000             # number of outputs
p, s, imp = np.zeros((noutputs, 3)), np.zeros((noutputs, 3)), np.zeros((noutputs, 3)) # position
vp, vs, vimp = np.zeros((noutputs, 3)), np.zeros((noutputs, 3)), np.zeros((noutputs, 3)) # velocity
dr, dv, h = np.zeros((noutputs, 3)), np.zeros((noutputs, 3)), np.zeros((noutputs, 3))
totaltime = t
times = np.linspace(0,totaltime, noutputs) # create times for integrations

for j in range(1):
    simp = rndm(10, 200, g=-1.6, size=1)*1e3 # impactor radius
    # simp = 100e3
    # print(simp)
    b = np.random.uniform(2,8)
    b = 2.6
    mimp = 4./3.*np.pi*dens*simp**3   # mass of impactor
    bhill = b*rhill # impact parameter
    theta = 0.0004  # true anomaly of impactor
    
    def setupSimulation():
        sim = rebound.Simulation()              # initialize rebound simulation
        sim.G = g                               # set G which sets units of integrator - SI in this case
        sim.collision = 'direct'
        sim.add(m=m1, r=s1, hash="primary")
        sim.add(m=m2, r=s2, a=a, e=e, inc=i, hash="secondary")
        sim.add(m=msun, a=rsun, f=np.pi, hash="sun")
        sim.move_to_com()
        sim.add(m=mimp, r=simp, a=rsun+bhill, f=theta, hash="impactor")
        return sim
    
    sim = setupSimulation()
    ps = sim.particles                      # create variable containing particles in simulation
    
    all_ps = [p.hash.value for j, p in enumerate(ps)]
    
    timer = timed() # start timer to time simulations
    try:
        for k, time in enumerate(times):
            sim.integrate(time)
            # print(k)
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
    # print(timed()-timer) # finish timer
    
    dr[:,0] = np.linalg.norm(p-s, axis=1)
    dr[:,1] = np.linalg.norm(p-imp, axis=1)
    dr[:,2] = np.linalg.norm(s-imp, axis=1)    
    dv[:,0] = np.linalg.norm(vp-vs, axis=1)
    dv[:,1] = np.linalg.norm(vp-vimp, axis=1)
    dv[:,2] = np.linalg.norm(vs-vimp, axis=1)
    h[:,0] = np.cross(p-s,vp-vs)[:,2]
    h[:,1] = np.cross(p-imp,vp-vimp)[:,2]
    h[:,2] = np.cross(s-imp,vs-vimp)[:,2]
    mu = g*(m1+m2)
    
    window = int(noutputs*pbin/totaltime*2) # set moving average window to be equal to 2 orbits of binary
    
    a = mu*dr/(2*mu - dr*dv**2)
    energy = -mu/2/a
    e = np.sqrt(1 + (2*energy*h**2 / mu**2))
    e_movingavg = np.zeros(noutputs)
    for ii in range(noutputs):
        if ii > window:
            xi = ii - window
        else:
            xi = 0
        e_movingavg[ii] = np.mean(e[xi:ii,0])
    
    # for ii in range(noutputs):
    #     e_cumsum = np.mean(e[ii-300:ii])
        # print(e_cumsum)
        
    e_rootsquaremean = np.sqrt(np.mean(e**2))
    a = a[-1]
    # print("e = " + str(e) + "\t" + "a = " + str(a/rhill))
    # print("a = " + str(a/rhill))
    # print("final eccentricity = " + str(e[-1]))

plt.figure(figsize=(10,6))
plt.plot(times/t,e[:,0])
plt.plot(times/t,e_movingavg)
plt.ylim(0,1)
plt.savefig(f"./img/ecc_moving_avg_test2", bbox_inches='tight')

# plt.figure()
# plt.plot(dr/rhill)
# plt.ylim(0,1)

omegak = np.sqrt(g*msun/rsun**3)
angles = -omegak*times
# sp = (s-p)/rhill         # difference between positions of secondary and primary
# ip = (imp-p)/rhill       # difference between positions of impactor and primary
# cosspx, cosspy = np.cos(angles)*sp[:,0], np.cos(angles)*sp[:,1] # cos of reference angles times difference between positions of secondary and primary
# sinspx, sinspy = np.sin(angles)*sp[:,0], np.sin(angles)*sp[:,1] # sin of reference angles times difference between positions of secondary and primary

# xdot = vs[:,0] - vp[:,0]        # x component of difference in velocities between secondary and primary
# ydot = vs[:,1] - vp[:,1]        # y component of difference in velocities between secondary and primary
# cosspxdot, cosspydot = np.cos(angles)*xdot, np.cos(angles)*ydot # cos of reference angles times difference between velocities of secondary and primary
# sinspxdot, sinspydot = np.sin(angles)*xdot, np.sin(angles)*ydot # sin of reference angles times difference between velocities of secondary and primary

# x, y = cosspx-sinspy+rsun, sinspx+cosspy                # x and y values for calculating jacobian constant
# vx, vy = cosspxdot-sinspydot, sinspxdot+cosspydot       # vx and vy values for calculating jacobian constant

# n = 2*np.pi/t                               # mean motion of binary around the sun

# cj = n**2*(x**2 + y**2) + 2*(mu/dr + mu/dr) - vx**2 - vy**2 # jacobian constant

ref = np.zeros((noutputs,3))
ref[:,0] = rsun*np.cos(angles)
ref[:,1] = 0 - rsun*np.sin(angles)

pref = (p-ref)/rhill
sref = (s-ref)/rhill
impref = (imp-ref)/rhill
cospx, cospy = np.cos(angles)*pref[:,0], np.cos(angles)*pref[:,1]
cossx, cossy = np.cos(angles)*sref[:,0], np.cos(angles)*sref[:,1]
cosix, cosiy = np.cos(angles)*impref[:,0], np.cos(angles)*impref[:,1]
sinpx, sinpy = np.sin(angles)*pref[:,0], np.sin(angles)*pref[:,1]
sinsx, sinsy = np.sin(angles)*sref[:,0], np.sin(angles)*sref[:,1]
sinix, siniy = np.sin(angles)*impref[:,0], np.sin(angles)*impref[:,1]
# %%
lim = 10
fig, axes = plt.subplots(1, figsize=(5, 5))
axes.set_xlabel("$x/R_\mathrm{H}$")
axes.set_ylabel("$y/R_\mathrm{H}$")
axes.set_ylim(-lim,lim)
axes.set_xlim(-lim,lim)

i = 999

Rhillprim = rsun*(m1/msun/3.)**(1./3.)/rhill
Rhillsec = rsun*(m2/msun/3.)**(1./3.)/rhill
Rhillimp = rsun*(mimp/msun/3.)**(1./3.)/rhill
# axes.grid()
primaryhill = plt.Circle((cospx[i]-sinpy[i], sinpx[i]+cospy[i]), Rhillprim, fc="none", ec="tab:orange", zorder=100)
axes.add_artist(primaryhill)
secondaryhill = plt.Circle((cossx[i]-sinsy[i], sinsx[i]+cossy[i]), Rhillsec, fc="none", ec="tab:blue")
axes.add_artist(secondaryhill)
impactorhill = plt.Circle((cosix[i]-siniy[i], sinix[i]+cosiy[i]), Rhillimp, fc="none", ec="tab:green")
axes.add_artist(impactorhill)
ms = 5
lw = 1
axes.plot(cospx[0:i]-sinpy[0:i], sinpx[0:i]+cospy[0:i], c="tab:orange", lw=lw)
axes.plot(cossx[0:i]-sinsy[0:i], sinsx[0:i]+cossy[0:i], c="tab:blue", lw=lw)
axes.plot(cosix[0:i]-siniy[0:i], sinix[0:i]+cosiy[0:i], c="tab:green", lw=lw)

axes.plot(cospx[i]-sinpy[i], sinpx[i]+cospy[i], c="tab:orange", marker='o', ms=ms, label="primary")
axes.plot(cossx[i]-sinsy[i], sinsx[i]+cossy[i], c="tab:blue", marker='o', ms=ms, label="secondary")
axes.plot(cosix[i]-siniy[i], sinix[i]+cosiy[i], c="tab:green", marker='o', ms=ms, label="impactor")
# axes.text(-4.5, -4.5, 't = {} Years'.format(int(times[i]/(year))), fontsize=12)

axes.legend()
# fig.savefig('./img/changes_test.pdf', bbox_inches='tight')
# %%
y = energy
plt.figure(figsize=(5,3))
# plt.title(f"Integrator={sim.integrator} -- Impactor radius={simp/1e3} km -- b={b/Rhill} hill radii")
plt.plot(times/year, y[:,0], label="prim-sec", lw=1)
plt.plot(times/year, y[:,1], label="prim-imp", lw=1)
plt.plot(times/year, y[:,2], label="sec-imp", lw=1)
plt.axhline(y=0, ls="--", color="black", lw=1)
plt.xlabel("Time (years)")
plt.ylabel("Energy (J/kg)")
plt.xlim(0, np.amax(times)/year)
# plt.ylim(-2, 30)
plt.grid('both')
plt.legend()
# plt.savefig(f"./img/single_energy2.pdf", bbox_inches='tight')
# %%
y = dr
plt.figure(figsize=(5,3))
# plt.title(f"Integrator={sim.integrator} -- Impactor radius={simp/1e3} km -- b={b/Rhill} hill radii")
plt.plot(times/year, y[:,0]/rhill, label="prim-sec", lw=1)
plt.plot(times/year, y[:,1]/rhill, label="prim-imp", lw=1)
plt.plot(times/year, y[:,2]/rhill, label="sec-imp", lw=1)
plt.xlabel("Time (years)")
plt.ylabel("Distance (Hill Radii)")
plt.xlim(0, np.amax(times)/year)
plt.ylim(0)
plt.grid('both')
plt.legend()
plt.savefig("./img/single_dr.pdf", bbox_inches='tight')
# %%
y = e_all
plt.figure(figsize=(5,3))
# plt.title(f"Integrator={sim.integrator} -- Impactor radius={simp/1e3} km -- b={b/Rhill} hill radii")
plt.plot(times/year, y[:,0], label="prim-sec", lw=1)
plt.plot(times/year, y[:,1], label="prim-imp", lw=1)
plt.plot(times/year, y[:,2], label="sec-imp", lw=1)
plt.xlabel("Time (years)")
plt.ylabel("eccentricity")
plt.xlim(0, np.amax(times)/year)
plt.ylim(0, 1)
plt.grid('both')
plt.legend()
plt.savefig(f"./img/single_eccentricity.pdf", bbox_inches='tight')
# %%
y = cj
plt.figure(figsize=(9,4))
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
lim = 10

fig, axes = plt.subplots(1, figsize=(7, 7))
axes.set_xlabel("$x/R_\mathrm{h}$")
axes.set_ylabel("$y/R_\mathrm{h}$")
axes.set_ylim(-lim,lim)
axes.set_xlim(-lim,lim)
primaryline, = axes.plot([], [], label="primary", c="tab:orange", lw=1.2)
secondaryline, = axes.plot([], [], label="secondary", c="tab:blue", lw=1)
impactorline, = axes.plot([], [], label="impactor", c="tab:green", lw=1.2)
primarydot, = axes.plot([], [], marker="o", ms=7, c="tab:orange")   
secondarydot, = axes.plot([], [], marker="o", ms=7, c="tab:blue")
impactordot, = axes.plot([], [], marker="o", ms=7, c="tab:green")
text = axes.text(-lim+(lim/10), lim-(lim/10), '', fontsize=15)
axes.legend()
axes.grid()

rhillprim = rsun*(m1/msun/3.)**(1./3.)/rhill
rhillsec = rsun*(m2/msun/3.)**(1./3.)/rhill
rhillimp = rsun*(mimp/msun/3.)**(1./3.)/rhill
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
f = f'videos/ps_animation_3.mp4' 
writervideo = FFMpegWriter(fps=100) # ffmpeg must be installed
anim.save(f, writer=writervideo)
# %%
lim = 1
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

anim = animation.FuncAnimation(fig, animate, frames=noutputs, interval=1)

# %%
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


# vorb = np.sqrt(g*(m1+m2)/rbin)              # orbital speed of primary and secondary around each other
# pbin = 2.*np.pi/np.sqrt(g*(m1+m2)/rbin**3)  # orbital period of primary and secondary around each other
# vk = np.sqrt(g*msun/rsun)      # orbital speed of binary around sun


# vshear = -1.5*omegak*rbin
# y0 = b*2                 # initial y distance of impactor from binary - larger for larger impactors
# r_a = rbin*(1-e)
# xb1 = -m2/(m1+m2)*r_a                  # slightly adjust initial x position of primary to keep centre of mass of binary at r
# xb2 = m1/(m1+m2)*r_a                   # slightly adjust initial x position of secondary to keep centre of mass of binary at r
# vshear1 = -1.5*omegak*xb1
# vshear2 = -1.5*omegak*xb2
# vorb = np.sqrt(g*(m1+m2)*(2/r_a-1/rbin))
# vorb1 = -m2/(m1+m2)*vorb                # orbital speed of primary around secondary - adjusted to account for offset from COM
# vorb2 = m1/(m1+m2)*vorb                 # orbital speed of secondary around primary - adjusted to account for offset from COM
# vorbi = np.sqrt(g*msun/(rsun+b))        # orbital speed of impactor around sun
# theta0 = y0/(rsun+b)                    # angle between impactor and line between binary COM and sun
# stheta0 = np.sin(theta0)                # sin of theta - needed for position of impactor
# ctheta0 = np.cos(theta0)                # cos of theta - needed for position of impactor
# impx = (rsun+b)*ctheta0-rsun  # x position of impactor
# impy = (rsun+b)*stheta0    # y position of impactor
# impvx = -vorbi*stheta0                  # x velocity of impactor
# impvy = vorbi*ctheta0                   # y velocity of impactor


    # sim.add(m=msun, hash="sun")
    # sim.add(m=m1, x=rsun+xb1, vy=vk+vorb1, hash="primary")
    # sim.add(m=m2, x=rsun+xb2, vy=vk+vorb2, hash="secondary")
    # sim.add(m=0, r=simp, x=impx+rsun, y=impy, vx=impvx, vy=impvy, hash="impactor")

    
    # secondaryline.set_data(s[0:i,0]/rhill-p[0:i,0]/rhill, s[0:i,1]/rhill-p[0:i,1]/rhill)
    # secondarydot.set_data(s[i,0]-p[i,0], s[i,1]-p[i,1])
    # impactorline.set_data(imp[0:i,0]-p[0:i,0], imp[0:i,1]-p[0:i,1])
    # impactordot.set_data(imp[i,0]-p[i,0], imp[i,1]-p[i,1])
    
    # primaryline.set_data(p[0:i,0], p[0:i,1])
    # secondaryline.set_data(s[0:i,0], s[0:i,1])
    # impactorline.set_data(imp[0:i,0], imp[0:i,1])