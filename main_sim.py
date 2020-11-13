# %%
import rebound
import numpy as np
import pandas as pd
from timeit import default_timer as timed


g = 6.67428e-11                             # gravitational constanct in SI units
au = 1.496e11                               # astronomical unit    
msun = 1.9891e30                            # mass of sun
s1, s2 = 100e3, 100e3                         # radius of primary and of secondary
dens1, dens2, densimp = 500., 500., 500. # density of primary, secondary, and impactor 
m1 = 4./3.*np.pi*dens1*s1**3                # mass of primary calculated from density and radius
m2 = 4./3.*np.pi*dens2*s2**3                # mass of secondary calculated from density and radius
rsun = 44.*au                                  # distance of centre of mass of binary from the sun 
rhill = rsun*(m1/msun/3.)**(1./3.)        # Hill radius of primary
rbin = 0.4*rhill                            # separation of binary
e = 0.5
r_a = rbin*(1+e)
vorb = np.sqrt(g*(m1+m2)/rbin)              # orbital speed of primary and secondary around each other
pbin = 2.*np.pi/np.sqrt(g*(m1+m2)/rbin**3)  # orbital period of primary and secondary around each other
t = 2.*np.pi/np.sqrt(g*msun/rsun**3)         # orbital period of binary around the sun
vk = np.sqrt(g*msun/rsun)

binaryi = np.deg2rad(0)     # inclination of binary
impi = np.deg2rad(0)        # inclination of impactor

noutputs = 1000             # number of outputs for plotting
p, s, imp = np.zeros((noutputs, 3)), np.zeros((noutputs, 3)), np.zeros((noutputs, 3))
vp, vs, vimp = np.zeros((noutputs, 3)), np.zeros((noutputs, 3)), np.zeros((noutputs, 3))

# header for pandas dataframe
headers = ['time','b','imp radius',
           'mass prim','x prim','y prim','z prim','vx prim','vy prim','vz prim',
           'mass sec','x sec','y sec','z sec','vx sec','vy sec','vz sec',
           'mass imp','x imp','y imp','z imp','vx imp','vy imp','vz imp',]
coll_headers = ['time','body','r','m','x','y','z','vx','vy','vz']

sim_name = "test_ecc"

simp = np.arange(100e3,101e3,10e3) # create range of impactor sizes to loop through
b = np.arange(3.0,3.1,0.2)*rhill # create range of impact parameters to loop through

timer = timed() # start timer to time simulations

for j in range(len(b)):             # loop through each impact parameter
    for i in range(len(simp)):      # loop throught each impactor radius
        mimp = 4./3.*np.pi*densimp*simp[i]**3   # mass of impactor
        xb1 = -m2/(m1+m2)*rbin                  # slightly adjust initial x position of primary to keep centre of mass of binary at r
        xb2 = m1/(m1+m2)*rbin                   # slightly adjust initial x position of secondary to keep centre of mass of binary at r

        vorb1 = -m2/(m1+m2)*vorb                # orbital speed of primary around secondary - adjusted to account for offset from COM
        vorb2 = m1/(m1+m2)*vorb                 # orbital speed of secondary around primary - adjusted to account for offset from COM
        sinbin = np.sin(binaryi)                # sin of inclination of binary
        cosbin = np.cos(binaryi)                # cos of inclination of binary
        
        primx = xb1*np.sin(np.pi/2-binaryi)     # x position of primary - accounts for inclination
        primz = xb1*sinbin             # z position of primary - accounts for inclination
        primvy = vk+vorb1*cosbin               # y velocity of primary - vy is keplerian velocity plus vorb
        primvz = -vorb1*sinbin                  # z velocity of primary - added if i > 0
        
        secx = xb2*np.sin(np.pi/2-binaryi)     # x position of secondary - accounts for inclination
        secz = xb2*sinbin             # z position of secondary - accounts for inclination
        secvy = vk+vorb2*cosbin               # y velocity of secondary - vy is keplerian velocity plus vorb
        secvz = -vorb2*sinbin                  # z velocity of secondary - added if i > 0
        
        y0 = rhill*simp[i]/s1*2.5 * s1/100e3  *b[j]/rhill               # initial y distance of impactor from binary - larger for larger impactors
        y0 = 10*rhill
        
        vorbi = np.sqrt(g*msun/(rsun+b[j]))        # orbital speed of impactor around sun
        theta0 = y0/(rsun+b[j])                    # angle between impactor and line between binary COM and sun
        stheta0 = np.sin(theta0)                # sin of theta - needed for position of impactor
        ctheta0 = np.cos(theta0)                # cos of theta - needed for position of impactor
        
        impx = (rsun+b[j])*ctheta0-rsun*np.cos(impi)  # x position of impactor
        impy = (rsun+b[j])*stheta0*np.cos(impi)    # y position of impactor
        impz = (rsun+b[j])*np.sin(impi)            # z position of impactor
        impvx = -vorbi*stheta0                  # x velocity of impactor
        impvy = vorbi*ctheta0                   # y velocity of impactor
        
        print('step ' + str(j + 1) + '-' + str(i+1))
        totaltime = t*y0/rhill/10 # total time of simulation - adjusted for different impactor sizes and distances
        totaltime = t
        # print(totaltime/t)
        times = np.reshape(np.linspace(0.,totaltime, noutputs), (noutputs,1)) # create times for integrations - reshape for hstack below
        
        def setupSimulation():
            sim = rebound.Simulation()              # initialize rebound simulation
            sim.G = g                               # set G which sets units of integrator - SI in this case
            sim.dt = 1e3                     # set initial timestep of integrator - IAS15 is adaptive so this will change
            sim.softening = 0.1*s1                  # softening parameter which modifies potential of each particle to prevent divergences
            sim.collision = 'direct'
            sim.add(m=msun, hash="sun")
            sim.add(m=m1, r=s1, x=rsun+primx, z=primz, vy=primvy, vz=primvz, hash="primary")
            sim.add(m=m2, r=s2, x=rsun+secx, z=secz, vy=secvy, vz=secvz, hash="secondary")
            sim.add(m=mimp, r=simp[i], x=rsun+impx, y=impy, z=impz, vx=impvx, vy=impvy, hash="impactor")
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
            collided = []
            for item in sim.particles:
                if item.lastcollision == sim.t:
                    collided.append([sim.t, item.index, item.r, item.m, item.x, item.y, item.z, item.vx, item.vy, item.vz])
            collided = np.array(collided)
            df_coll = pd.DataFrame(collided)
            df_coll.to_csv(f'./results/collision_{sim_name}_b-{np.round(b[j]/rhill, 1)}_r-{np.round(simp[i]/1e3, 1)}.csv', header=coll_headers)
 
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
            
        # create matrix of results in same order as header created above - reshape some to avoid error
        particles = np.hstack((times,
                               np.reshape(np.ones(noutputs)*b[j]/rhill,(noutputs,1)),
                               np.reshape(np.ones(noutputs)*simp[i]/1e3,(noutputs,1)),
                               np.reshape(np.ones(noutputs)*m1, (noutputs,1)),
                               p,
                               vp,
                               np.reshape(np.ones(noutputs)*m2, (noutputs,1)),
                               s,
                               vs,
                               np.reshape(np.ones(noutputs)*mimp,(noutputs,1)),
                               imp,
                               vimp))
        
        df = pd.DataFrame(particles)
        # write to csv with impactor size and impact parameter in title - round values to avoid long file names
        df.to_csv(f'./results/{sim_name}_b-{np.round(b[j]/rhill, 1)}_r-{np.round(simp[i]/1e3, 1)}.csv', header=headers)
            
print(timed()-timer) # finish timer