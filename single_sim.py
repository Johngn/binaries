import numpy as np
import rebound
import matplotlib.pyplot as plt
import matplotlib as matplotlib
matplotlib.rcParams.update({'figure.max_open_warning': 0})
import mpl_toolkits.mplot3d.axes3d as p3
from powerlaw import rndm
from scipy.stats import powerlaw
from scipy.optimize import fsolve
from timeit import default_timer as timed
from matplotlib import animation
from matplotlib.animation import FuncAnimation, FFMpegWriter

g = 6.67428e-11                             # gravitational constant in SI units
au = 1.496e11                               # astronomical unit    
msun = 1.9891e30                            # mass of sun
rsun = 44.*au                                  # distance from centre of mass of binary to sun 
year = 365.25*24.*60.*60.                   # number of seconds in a year
dens = 700.

s1, s2 = 100e3, 100e3                         # radius of primary and of secondary
m1 = 4./3.*np.pi*dens*s1**3                # mass of primary calculated from density and radius
m2 = 4./3.*np.pi*dens*s2**3                # mass of secondary calculated from density and radius
rhill = rsun*(m1/msun/3.)**(1./3.)        # Hill radius of primary

a = 0.2*rhill                            # separation of binary
e = 0
inc = np.deg2rad(0)

pbin = 2.*np.pi/np.sqrt(g*(m1+m2)/a**3)            # orbital period of binary around the sun
t = 2.*np.pi/np.sqrt(g*msun/rsun**3)            # orbital period of binary around the sun
noutputs = 100            # number of outputs
p, s, imp = np.zeros((noutputs, 3)), np.zeros((noutputs, 3)), np.zeros((noutputs, 3)) # position
vp, vs, vimp = np.zeros((noutputs, 3)), np.zeros((noutputs, 3)), np.zeros((noutputs, 3)) # velocity
totaltime = t*1
times = np.linspace(0,totaltime, noutputs) # create times for integrations

f = 0
omega = 0
Omega = 0

timer = timed() # start timer to time simulations
collision_totals = 0

a_total = []
e_total = []
inc_total = []

n_encounters = 2000

sim_name = "single_sim_highecc_4"
# lim = 10
for j in range(n_encounters):
    # print(j)
    simp = rndm(10, 200, g=-1.6, size=1)*1e3    # impactor radius
    b = np.random.uniform(-8,8)*rhill
    inc_imp = np.random.uniform(0,np.deg2rad(5))
    e_imp = np.random.uniform(0,0.5)

    
    # simp = 100e3
    # b = -5*rhill                          # impact parameter
    mimp = 4./3.*np.pi*dens*simp**3
    n_bin = np.sqrt(g*msun/rsun)**3        # mean motion of binary COM
    n_imp = np.sqrt(g*msun/(rsun+b))**3    # mean motion of impactor
    
    
    # f_enc   = np.arccos((rsun+b)*(1-e_imp**2)/(e_imp*(rsun+b)) - 1/e_imp)  # angle at which close encounter occurs
    f_enc   = np.arccos(-e_imp)  # always needs to be in second quadrant
    # E       = np.arctan( np.sqrt(1-e_imp**2) * np.sin(f_enc) / (e_imp + np.cos(f_enc)) ) # E at encounter    
    E       = np.arctan2( np.sqrt(1-e_imp**2) * np.sin(f_enc), e_imp + np.cos(f_enc))
    M       = E - e_imp*np.sin(E)           # mean anomaly at encounter
    t_enc   = np.pi/n_bin                   # time it takes for binary COM to get to crossover point
    M_0     = n_imp*-t_enc + M              # mean anomaly at start of sim
    
    def func(x):
        return x-e_imp*np.sin(x) - M_0
    
    E_0 = fsolve(func, 1)
    
    # f_0 = np.arccos( (np.cos(E_0) - e_imp) / (1 - e_imp*np.cos(E_0)) )
    
    f_0 = 2 * np.arctan2( np.sqrt(1+e_imp)*np.sin(E_0/2) , np.sqrt(1-e_imp)*np.cos(E_0/2) )
    
    # print(f_0)
    
    timer = timed() # start timer to time simulations

    def setupSimulation():
        sim = rebound.Simulation()              # initialize rebound simulation
        sim.G = g                               # set G which sets units of integrator - SI in this case
        sim.collision = 'direct'
        sim.add(m=m1, r=s1, hash="primary")
        sim.add(m=m2, r=s2, a=a, e=e, omega=omega, f=f, inc=inc, Omega=Omega, hash="secondary")
        sim.add(m=msun, a=rsun, f=np.pi, hash="sun")
        sim.move_to_com()
        sim.add(m=mimp, r=simp, a=rsun+b, e=e_imp, omega=np.pi-f_enc, f=f_0, inc=inc_imp, Omega=0, hash="impactor")
        return sim
    
    sim = setupSimulation()
    ps = sim.particles                      # create variable containing particles in simulation
    
    all_ps = [p.hash.value for j, p in enumerate(ps)]
    
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
        print("collision detected")
        collision = True
        collision_totals =+ 1
        collided = []
        for item in sim.particles:
            if item.lastcollision == sim.t:
                collided.append([sim.t, item.index, item.r, item.m, item.x, item.y, item.z, item.vx, item.vy, item.vz])
        collided = np.array(collided)
        
        if(np.array_equal(collided[:,1], [0,1])):
            print("binary collided")
        
        sim.collision_resolve = 'merge'
    
        for k, time in enumerate(times):
            sim.integrate(time)
            existing_ps = [p.hash.value for j, p in enumerate(ps)]
            if all_ps[0] in existing_ps:
                p[k] = [ps["primary"].x, ps["primary"].y, ps["primary"].z]
                vp[k] = [ps["primary"].vx, ps["primary"].vy, ps["primary"].vz]
            if all_ps[1] in existing_ps:
                s[k] = [ps["secondary"].x, ps["secondary"].y, ps["secondary"].z]
                vs[k] = [ps["secondary"].vx, ps["secondary"].vy, ps["secondary"].vz]
            if all_ps[3] in existing_ps:
                imp[k] = [ps["impactor"].x, ps["impactor"].y, ps["impactor"].z]
                vimp[k] = [ps["impactor"].vx, ps["impactor"].vy, ps["impactor"].vz]
                
      
        m1 = sim.particles[0].m
        m1 = sim.particles[1].m
        
    # print(timed()-timer) # finish timer
                
    orbit = sim.particles[1].calculate_orbit(sim.particles[0])
    a = orbit.a
    e = orbit.e
    vk_bin  = np.sqrt(g*msun/rsun)               # initial orbital speed at binary COM
    vk_imp  = np.sqrt(g*msun/(rsun+b))       # initial orbital speed at impactor
    inc = orbit.inc
    Omega = orbit.Omega
    omega = orbit.omega
    f = orbit.f
    

    
    R, V, mu, hz, h_mag = np.zeros((noutputs,3)), np.zeros((noutputs,3)), np.zeros((noutputs,3)), np.zeros((noutputs,3)), np.zeros((noutputs,3))
    R[:,0] = np.linalg.norm(p-s, axis=1)
    R[:,1] = np.linalg.norm(p-imp, axis=1)
    R[:,2] = np.linalg.norm(s-imp, axis=1)
    V[:,0] = np.linalg.norm(vp-vs, axis=1)
    V[:,1] = np.linalg.norm(vp-vimp, axis=1)
    V[:,2] = np.linalg.norm(vs-vimp, axis=1)
    hz[:,0] = np.cross(p-s,vp-vs)[:,2]
    hz[:,1] = np.cross(p-imp,vp-vimp)[:,2]
    hz[:,2] = np.cross(s-imp,vs-vimp)[:,2]
    h_mag[:,0] = np.linalg.norm(np.cross(p-s,vp-vs), axis=1)
    h_mag[:,1] = np.linalg.norm(np.cross(p-imp,vp-vimp), axis=1)
    h_mag[:,2] = np.linalg.norm(np.cross(s-imp,vs-vimp), axis=1)
    mu[:,0] = g*(m1+m2)
    mu[:,1] = g*(m1+mimp)
    mu[:,2] = g*(m2+mimp)

    Rhill = np.array([rsun*(m1/msun/3.)**(1./3.), rsun*(m2/msun/3.)**(1./3.), rsun*(mimp/msun/3.)**(1./3.)])
    Rhill_largest = np.array([np.amax([Rhill[0], Rhill[1]]), np.amax([Rhill[0], Rhill[2]]), np.amax([Rhill[1], Rhill[2]])])
    
    a_final = mu*R/(2*mu - R*V**2)
    energy = -mu/2/a_final
    e_final = np.sqrt(1 + (2*energy*h_mag**2 / mu**2))
    inc_final = np.arccos(hz/h_mag)
    
    e_total.append(e_final[:,0])
    a_total.append(a_final[:,0])
    inc_total.append(inc_final[:,0])
    
    bound = np.logical_and(np.logical_and(energy < 0, np.isfinite(energy)), R < Rhill_largest)[-1]
    
    omegak = np.sqrt(g*msun/rsun**3)
    angles = -omegak*times
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
    
a_total = np.array(a_total).flatten()/rhill
e_total = np.array(e_total).flatten()
inc_total = np.array(inc_total).flatten()

a_total = np.reshape(a_total, (noutputs*n_encounters, 1))
e_total = np.reshape(e_total, (noutputs*n_encounters, 1))
inc_total = np.reshape(inc_total, (noutputs*n_encounters, 1))

save_data = np.hstack((a_total, e_total, inc_total))

np.savetxt(f"./data/{sim_name}", save_data)
    
    # %%
    lim = 8
    fig, axes = plt.subplots(1, figsize=(5,5))
    axes.set_xlabel("$x/R_\mathrm{H}$")
    axes.set_ylabel("$y/R_\mathrm{H}$")
    axes.set_ylim(-lim,lim)
    axes.set_xlim(-lim,lim)
    
    i = -120
    ii = i
    
    color1 = "teal"
    color2 = "hotpink"
    color3 = "sienna"
    
    Rhillprim = rsun*(m1/msun/3.)**(1./3.)/rhill
    Rhillsec = rsun*(m2/msun/3.)**(1./3.)/rhill
    Rhillimp = rsun*(mimp/msun/3.)**(1./3.)/rhill
    # axes.grid()
    primaryhill = plt.Circle((cospx[i]-sinpy[i], sinpx[i]+cospy[i]), Rhillprim, fc="none", ec=color1)
    axes.add_artist(primaryhill)
    secondaryhill = plt.Circle((cossx[i]-sinsy[i], sinsx[i]+cossy[i]), Rhillsec, fc="none", ec=color2)
    axes.add_artist(secondaryhill)
    impactorhill = plt.Circle((cosix[i]-siniy[i], sinix[i]+cosiy[i]), Rhillimp, fc="none", ec=color3)
    axes.add_artist(impactorhill)
    lw = 1.2
    ms = 5
    axes.plot(cospx[0:i]-sinpy[0:i], sinpx[0:i]+cospy[0:i], c=color1, lw=lw)
    axes.plot(cossx[0:i]-sinsy[0:i], sinsx[0:i]+cossy[0:i], c=color2, lw=lw)
    axes.plot(cosix[0:ii]-siniy[0:ii], sinix[0:ii]+cosiy[0:ii], c=color3, lw=lw)
    axes.plot(cospx[i]-sinpy[i], sinpx[i]+cospy[i], c=color1, marker='o', ms=ms, label="primary")
    axes.plot(cossx[i]-sinsy[i], sinsx[i]+cossy[i], c=color2, marker='o', ms=ms, label="secondary")
    axes.plot(cosix[i]-siniy[i], sinix[i]+cosiy[i], c=color3, marker='o', ms=ms, label="impactor")
    # axes.text(-4.5, -4.5, 't = {} Years'.format(int(times[i]/(year))), fontsize=12)
    
    # axes.grid()
    axes.legend()
    # fig.savefig(f'./img/chaotic_encounters_e_2.pdf', bbox_inches='tight')
    
        # %%
    

lim = 10

color1 = "teal"
color2 = "hotpink"
color3 = "sienna"
lw = 1.5
ms = 5

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
axes.legend()
axes.grid()

rhillprim = rsun*(m1/msun/3.)**(1./3.)/rhill
rhillsec = rsun*(m2/msun/3.)**(1./3.)/rhill
rhillimp = rsun*(mimp/msun/3.)**(1./3.)/rhill
primaryhill = plt.Circle((0,0), rhillprim, fc="none", ec=color1)    
secondaryhill = plt.Circle((0,0), rhillsec, fc="none", ec=color2)
impactorhill = plt.Circle((0,0), rhillimp, fc="none", ec=color3)

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

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=noutputs, interval=1)
# %%
    color1 = "teal"
    color2 = "hotpink"
    color3 = "sienna"
    lim = 44*au
    fig = plt.figure(figsize=(10,10))
    axes = fig.add_subplot(111, projection='3d')
    axes.set_xlabel("$x/R_\mathrm{h}$") 
    axes.set_ylabel("$y/R_\mathrm{h}$")
    axes.set_zlabel("$z/R_\mathrm{h}$") 
    axes.set_xlim3d([-lim, lim])
    axes.set_ylim3d([-lim, lim])
    axes.set_zlim3d([-lim, lim])
    primaryline, = axes.plot([], [], [], label="primary", c=color1, lw=3.5)
    secondaryline, = axes.plot([], [], [], label="secondary", c=color2, lw=1.5)
    impactorline, = axes.plot([], [], [], label="impactor", c=color3, lw=1.5)
    primarydot, = axes.plot([], [], [], marker="o", ms=8, c=color1)
    secondarydot, = axes.plot([], [], [], marker="o", ms=8, c=color2)
    impactordot, = axes.plot([], [], [], marker="o", ms=8, c=color3)
    text = axes.text(-lim+(lim/10), lim-(lim/10), lim-(lim/10), '', fontsize=15)
    axes.legend()
    
    def animate(i):
        primaryline.set_data(p[0:i,0], p[0:i,1])
        secondaryline.set_data(s[0:i,0], s[0:i,1])  
        impactorline.set_data(imp[0:i,0], imp[0:i,1])
        primaryline.set_3d_properties(p[0:i,2])
        secondaryline.set_3d_properties(s[0:i,2])
        impactorline.set_3d_properties(imp[0:i,2])
        primarydot.set_data(p[i,0], p[i,1])
        secondarydot.set_data(s[i,0], s[i,1])
        impactordot.set_data(imp[i,0], imp[i,1])
        primarydot.set_3d_properties(p[i,2])
        secondarydot.set_3d_properties(s[i,2])
        impactordot.set_3d_properties(imp[i,2])
        text.set_text('{} Years'.format(int(times[i]/(year))))
        return primarydot, secondarydot, impactordot, primaryline, secondaryline, impactorline, text
    
    anim = animation.FuncAnimation(fig, animate, frames=noutputs, interval=1)
    
    # %%

# %%

    i = -1
    color1 = "teal"
    color2 = "hotpink"
    color3 = "sienna"
    lw = 1.2
    ms = np.arccos((rsun+b)*(1-e_imp**2)/(e_imp*(rsun+b)) - 1/e_imp)
    lim = 40*au
    fig = plt.figure(figsize=(10,10))
    axes = fig.add_subplot(111, projection='3d')
    axes.set_xlabel("$x/R_\mathrm{h}$")
    axes.set_ylabel("$y/R_\mathrm{h}$")
    axes.set_zlabel("$z/R_\mathrm{h}$") 
    axes.set_xlim3d([-lim, lim])    
    axes.set_ylim3d([-lim, lim])
    axes.set_zlim3d([-lim, lim])
    
    axes.plot(p[0:i,0], p[0:i,1], p[0:i,2], c=color1, lw=lw*2)
    axes.plot(s[0:i,0], s[0:i,1], s[0:i,2], c=color2, lw=lw)
    axes.plot(imp[0:i,0], imp[0:i,1], imp[0:i,2], c=color3, lw=lw)
    axes.scatter(p[i,0], p[i,1], p[i,2], c=color1, s=70, label="primary")
    axes.scatter(s[i,0], s[i,1], s[i,2], c=color2, s=50, label="secondary")
    axes.scatter(imp[i,0], imp[i,1], imp[i,2], c=color3, s=50, label="impactor")
    axes.scatter(0,0,0, c='gold', s=100)
    axes.legend()


    # %%

    color1 = "teal"
    color2 = "hotpink"
    color3 = "sienna"
    lim = 2
    fig = plt.figure(figsize=(8,8))
    axes = fig.add_subplot(111, projection='3d')
    axes.set_xlabel("$x/R_\mathrm{h}$")
    axes.set_ylabel("$y/R_\mathrm{h}$")
    axes.set_zlabel("$z/R_\mathrm{h}$") 
    axes.set_xlim3d([-lim, lim])
    axes.set_ylim3d([-lim, lim])
    axes.set_zlim3d([-lim, lim])
    
    axes.plot(cospx[0:i]-sinpy[0:i], sinpx[0:i]+cospy[0:i], pref[0:i,2], c=color1, lw=lw)
    axes.plot(cossx[0:i]-sinsy[0:i], sinsx[0:i]+cossy[0:i], sref[0:i,2], c=color2, lw=lw)
    axes.plot(cosix[0:i]-siniy[0:i], sinix[0:i]+cosiy[0:i], impref[0:i,2], c=color3, lw=lw)
    axes.scatter(cospx[i]-sinpy[i], sinpx[i]+cospy[i], pref[i,2], c=color1, s=50, label="primary")
    axes.scatter(cossx[i]-sinsy[i], sinsx[i]+cossy[i], sref[i,2], c=color2, s=50, label="secondary")
    axes.scatter(cosix[i]-siniy[i], sinix[i]+cosiy[i], impref[i,2], c=color3, s=50, label="impactor")
        
    axes.legend()


# %%


# %% 


# %%
plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
f = f'videos/ps_animation_3.mp4' 
writervideo = FFMpegWriter(fps=100) # ffmpeg must be installed
anim.save(f, writer=writervideo)

# %%
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
plt.ylim(-2, 30)
plt.grid('both')
plt.legend()
plt.savefig(f"./img/changes_energy2.pdf", bbox_inches='tight')
# %%
y = R
plt.figure(figsize=(5,3))
# plt.title(f"Integrator={sim.integrator} -- Impactor radius={simp/1e3} km -- b={b/Rhill} hill radii")
plt.plot(times/year, y[:,0]/rhill, label="prim-sec", lw=1)
color2 = "hotpink"
color3 = "sienna"
plt.plot(times/year, y[:,1]/rhill, label="prim-imp", lw=1)
plt.plot(times/year, y[:,2]/rhill, label="sec-imp", lw=1)
plt.xlabel("Time (years)")
plt.ylabel("Distance (Hill Radii)")
plt.xlim(0, np.amax(times)/year)
plt.ylim(0, 20)
plt.grid('both')
plt.legend()
plt.savefig("./img/changes_dr2.pdf", bbox_inches='tight')
# %%
y = e_total
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
# plt.savefig(f"./img/changes_eccentricity2.pdf", bbox_inches='tight')
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
color1 = "teal"
color2 = "hotpink"
color3 = "sienna"
lim = 10
    
fig = plt.figure(figsize=(10,10))
axes = fig.add_subplot(111, projection='3d')
axes.set_xlabel("$x/R_\mathrm{h}$")
axes.set_ylabel("$y/R_\mathrm{h}$")
axes.set_zlabel("$z/R_\mathrm{h}$")
axes.set_xlim3d([-lim, lim])
axes.set_ylim3d([-lim, lim])
axes.set_zlim3d([-lim, lim])
primaryline, = axes.plot([], [], [], label="primary", c=color1, lw=1.5)
secondaryline, = axes.plot([], [], [], label="secondary", c=color2, lw=1.5)
impactorline, = axes.plot([], [], [], label="impactor", c=color3, lw=1.5)
primarydot, = axes.plot([], [], [], marker="o", ms=8, c=color1)
secondarydot, = axes.plot([], [], [], marker="o", ms=8, c=color2)
impactordot, = axes.plot([], [], [], marker="o", ms=8, c=color3)
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

anim = animation.FuncAnimation(fig, animate, frames=noutputs, interval=1    )

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