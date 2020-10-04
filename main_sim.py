# %%
import glob, os, csv, rebound
import numpy as np
import pandas as pd
from timeit import default_timer as timed


G = 6.67428e-11                             # gravitational constanct in SI units
au = 1.496e11                               # astronomical unit    
Msun = 1.9891e30                            # mass of sun
year = 365.25*24.*60.*60.                   # number of seconds in a year
s1, s2 = 130e3, 80e3                         # radius of primary and of secondary
dens1, dens2, densimp = 1000., 1000., 1000. # density of primary, secondary, and impactor 
m1 = 4./3.*np.pi*dens1*s1**3                # mass of primary calculated from density and radius
m2 = 4./3.*np.pi*dens2*s2**3                # mass of secondary calculated from density and radius
rsun = 44.*au                                  # distance of centre of mass of binary from the sun 
OmegaK = np.sqrt(G*Msun/rsun**3)       # keplerian frequency at this distance
Rhill = rsun*(m1/Msun/3.)**(1./3.)        # Hill radius of binary
rbin = 0.3*Rhill                            # separation of binary is 0.5 of the Hill radius
vorb = np.sqrt(G*(m1+m2)/rbin)              # orbital speed of primary and secondary around each other
vshear = -1.5*OmegaK*rbin                   # calculates the change in velocity required to keep a body in a circular orbit
Pbin = 2.*np.pi/np.sqrt(G*(m1+m2)/rbin**3)  # orbital period of primary and secondary around each other
T = 2.*np.pi/np.sqrt(G*(Msun)/rsun**3)         # orbital period of binary around the sun
n = 2*np.pi/T                               # mean motion of binary around the sun
mu1 = G*Msun                                # mu of a body is G times its mass          
mu2 = G*m1

binaryi = np.deg2rad(0)     # inclination of binary
impi = np.deg2rad(0)        # inclination of impactor

Noutputs = 1000             # number of outputs for plotting
p, s, imp, sun = np.zeros((Noutputs, 3)), np.zeros((Noutputs, 3)), np.zeros((Noutputs, 3)), np.zeros((Noutputs, 3))
vp, vs, vimp, vsun = np.zeros((Noutputs, 3)), np.zeros((Noutputs, 3)), np.zeros((Noutputs, 3)), np.zeros((Noutputs, 3))
initial, final = [], [] # empty arrays for initial and final positions and velocities for each body

# header for pandas dataframe
headers = ['time','b','imp radius','mass prim','x prim','y prim','z prim','vx prim','vy prim','vz prim',
           'mass sec','x sec','y sec','z sec','vx sec','vy sec','vz sec',
           'mass imp','x imp','y imp','z imp','vx imp','vy imp','vz imp',
           'mass sun','x sun','y sun','z sun','vx sun','vy sun','vz sun',]
coll_headers = ['time','body','r','m','x','y','z','vx','vy','vz']

sim_name = "coord_test3"

simp = np.arange(101e3,200e3,20e3) # create range of impactor sizes to loop through
b = np.arange(2.3,6.5,1.4)*Rhill # create range of impact parameters to loop through

timer = timed() # start timer to time simulations

for j in range(len(b)):             # loop through each impact parameter
    for i in range(len(simp)):      # loop throught each impactor radius
        print('step ' + str(j + 1) + '-' + str(i+1))
        totaltime = 3*T*simp[i]/s1/b[j]*Rhill # total time of simulation - adjusted for different impactor sizes and distances
        print(totaltime/T)
        # totaltime = T/10
        times = np.reshape(np.linspace(0.,totaltime, Noutputs), (Noutputs,1)) # create times for integrations - reshape for hstack below
        y0 = Rhill*simp[i]/s1*10                 # initial y distance of impactor from binary - larger for larger impactors
        # y0 = Rhill
        mimp = 4./3.*np.pi*densimp*simp[i]**3   # mass of impactor
        xb1 = -m2/(m1+m2)*rbin                  # slightly adjust initial x position of primary to keep centre of mass of binary at r
        xb2 = m1/(m1+m2)*rbin                   # slightly adjust initial x position of secondary to keep centre of mass of binary at r
        
        vshear1 = -1.5*OmegaK*xb1               # keplerian shear of primary
        vshear2 = -1.5*OmegaK*xb2               # keplerian shear of secondary
        
        vK1 = np.sqrt(G*(Msun+m1)/(rsun+xb1))      # orbital speed of primary around sun
        vK2 = np.sqrt(G*(Msun+m2)/(rsun+xb2))      # inital orbital speed of secondary around sun
        
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
        
        vorbi = np.sqrt(G*Msun/(rsun+b[j]))        # orbital speed of impactor around sun
        theta0 = y0/(rsun+b[j])                    # angle between impactor and line between binary COM and sun
        stheta0 = np.sin(theta0)                # sin of theta - needed for position of impactor
        ctheta0 = np.cos(theta0)                # cos of theta - needed for position of impactor
        
        impx = (rsun+b[j])*ctheta0-rsun*np.cos(impi)  # x position of impactor
        impy = (rsun+b[j])*stheta0*np.cos(impi)    # y position of impactor
        impz = (rsun+b[j])*np.sin(impi)            # z position of impactor
        impvx = -vorbi*stheta0                  # x velocity of impactor
        impvy = vorbi*ctheta0                   # y velocity of impactor
        
        def setupSimulation():
            sim = rebound.Simulation()              # initialize rebound simulation
            sim.G = G                               # set G which sets units of integrator - SI in this case
            sim.dt = 1e-4*Pbin                      # set initial timestep of integrator - IAS15 is adaptive so this will change
            sim.softening = 0.1*s1                  # softening parameter which modifies potential of each particle to prevent divergences
            sim.collision = 'direct'
            # sim.collision_resolve = 'merge'
            sim.add(m=Msun, hash="sun")
            sim.add(m=m1, r=s1, x=rsun+primx, z=primz, vy=primvy, vz=primvz, hash="primary")
            sim.add(m=m2, r=s2, x=rsun+secx, z=secz, vy=secvy, vz=secvz, hash="secondary")
            sim.add(m=mimp, r=simp[i], x=rsun+impx, y=impy, z=impz, vx=impvx, vy=impvy, hash="impactor")
            return sim
        
        sim = setupSimulation()
        ps = sim.particles                      # create variable containing particles in simulation                
        
        
        for k, time in enumerate(times):
            try:
                sim.integrate(time)
                # print(k)
                # if ps['sun']
                sun[k] = [ps["sun"].x, ps["sun"].y, ps["sun"].z]
                p[k] = [ps["primary"].x, ps["primary"].y, ps["primary"].z]
                s[k] = [ps["secondary"].x, ps["secondary"].y, ps["secondary"].z]
                imp[k] = [ps["impactor"].x, ps["impactor"].y, ps["impactor"].z]
                vsun[k] = [ps["sun"].vx, ps["sun"].vy, ps["sun"].vz]
                vp[k] = [ps["primary"].vx, ps["primary"].vy, ps["primary"].vz]
                vs[k] = [ps["secondary"].vx, ps["secondary"].vy, ps["secondary"].vz]
                vimp[k] = [ps["impactor"].vx, ps["impactor"].vy, ps["impactor"].vz]
            except rebound.Collision:
                # print('error'+str(k))
                collided = []
                for item in sim.particles:
                    if item.lastcollision == sim.t:
                        collided.append([sim.t, item.index, item.r, item.m, item.x, item.y, item.z, item.vx, item.vy, item.vz])
                collided = np.array(collided)
                df_coll = pd.DataFrame(collided)
                df_coll.to_csv(f'./results/collision_{sim_name}__b-{np.round(b[j]/Rhill, 1)}__r-{np.round(simp[i]/1e3, 1)}.csv', header=coll_headers)
        
            
        # create matrix of results in same order as header created above - reshape some to avoid error
        particles = np.hstack((times,
                               np.reshape(np.ones(Noutputs)*b[j]/Rhill,(Noutputs,1)),
                               np.reshape(np.ones(Noutputs)*simp[i]/1e3,(Noutputs,1)),
                               np.reshape(np.ones(Noutputs)*m1, (Noutputs,1)),
                               p,
                               vp,
                               np.reshape(np.ones(Noutputs)*m2, (Noutputs,1)),
                               s,
                               vs,
                               np.reshape(np.ones(Noutputs)*mimp,(Noutputs,1)),
                               imp,
                               vimp,
                               np.reshape(np.ones(Noutputs)*Msun, (Noutputs,1)),
                               sun,
                               vsun))
        
        df = pd.DataFrame(particles)
        # write to csv with impactor size and impact parameter in title - round values to avoid long file names
        df.to_csv(f'./results/{sim_name}__b-{np.round(b[j]/Rhill, 1)}__r-{np.round(simp[i]/1e3, 1)}.csv', header=headers)
        
        initial.append(particles[0])    # initial positions and velocities of bodies
        final.append(particles[-1])     # final positions and velocities of bodies
            
print(timed()-timer) # finish timer

df = pd.DataFrame(initial)                              # create dataframe of initial values
df.to_csv(f'./results/{sim_name}_initial.csv', header=headers)     # write initial values to csv
df = pd.DataFrame(final)                                # create dataframe of final values
df.to_csv(f'./results/{sim_name}_final.csv', header=headers)       # write final values to csv