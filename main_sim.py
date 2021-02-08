# %%
import rebound
import numpy as np
import pandas as pd
from timeit import default_timer as timed


g = 6.67428e-11                             # gravitational constanct in SI units
au = 1.496e11                               # astronomical unit    
msun = 1.9891e30                            # mass of sun
s1, s2 = 100e3, 100e3                         # radius of primary and of secondary
dens = 700. # density of primary, secondary, and impactor 
m1 = 4./3.*np.pi*dens*s1**3                # mass of primary calculated from density and radius
m2 = 4./3.*np.pi*dens*s2**3                # mass of secondary calculated from density and radius
rsun = 44.*au                                  # distance of centre of mass of binary from the sun 
rhill = rsun*(m1/msun/3.)**(1./3.)         # Hill radius of primary

a = 0.4*rhill                            # separation of binary
e = 0

t = 2.*np.pi/np.sqrt(g*msun/rsun**3)         # orbital period of binary around the sun
totaltime = t*2
noutputs = 1000             # number of outputs for plotting
times = np.linspace(0,totaltime, noutputs) # create times for integrations
p, s, imp = np.zeros((noutputs, 3)), np.zeros((noutputs, 3)), np.zeros((noutputs, 3))
vp, vs, vimp = np.zeros((noutputs, 3)), np.zeros((noutputs, 3)), np.zeros((noutputs, 3))

# header for pandas dataframe
headers = ['time','b',
           'hash prim','mass prim','radius prim','x prim','y prim','z prim','vx prim','vy prim','vz prim',
           'hash sec','mass sec','radius sec','x sec','y sec','z sec','vx sec','vy sec','vz sec',
           'hash imp','mass imp','radius imp','x imp','y imp','z imp','vx imp','vy imp','vz imp',]

coll_headers = ['time','body','r','m','x','y','z','vx','vy','vz']

sim_name = "single_random_test"

# simp = np.arange(50e3,210e3,10e3) # create range of impactor sizes to loop through
# b = np.arange(2,4.1,0.2) # create range of impact parameters to loop through

simp = np.ones(10)*100e3
b = np.ones(1)*2.5


timer = timed() # start timer to time simulations

for j in range(len(b)):             # loop through each impact parameter
    for i in range(len(simp)):      # loop throught each impactor radius
        print('step ' + str(j + 1) + '-' + str(i+1))
        bhill = b[j]*rhill # impact parameter
        mimp = 4./3.*np.pi*dens*simp[i]**3   # mass of impactor
        theta = 0.0015  # true anomaly of impactor
        
        starting_position = np.round(np.random.uniform()*2*np.pi, 4)
        
        def setupSimulation():
            sim = rebound.Simulation()              # initialize rebound simulation
            sim.G = g                               # set G which sets units of integrator - SI in this case
            sim.collision = 'direct'
            sim.add(m=m1, r=s1, hash="primary")
            sim.add(m=m2, r=s2, a=a, e=e, f=starting_position, hash="secondary")
            sim.add(m=msun, a=rsun, f=np.pi, hash="sun")
            sim.move_to_com()
            sim.add(m=mimp, r=simp[i], a=rsun+bhill, f=theta, hash="impactor")
            return sim
            
        sim = setupSimulation()
        ps = sim.particles                      # create variable containing particles in simulation
        all_ps = [p.hash.value for j, p in enumerate(ps)]
        
        ps1 = ps["primary"].index
        ps2 = ps["secondary"].index
        ps3 = ps["impactor"].index
        
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
            print('collision detected')
            collided = []
            for item in sim.particles:
                if item.lastcollision == sim.t:
                    collided.append([sim.t, item.index, item.r, item.m, item.x, item.y, item.z, item.vx, item.vy, item.vz])
            collided = np.array(collided) 
            df_coll = pd.DataFrame(collided)
            # df_coll.to_csv(f'./results/collision_{sim_name}_{np.round(simp[i]/1e3, 1)}_{np.round(b[j], 1)}.csv', header=coll_headers)
            df_coll.to_csv(f'./results/collision_{sim_name}_{starting_position}.csv', header=coll_headers)
            
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
                    
            
        # create matrix of results in same order as header created above - reshape some to avoid error
        particles = np.hstack((np.reshape(times,(noutputs,1)),
                               np.reshape(np.ones(noutputs)*b[j]/rhill,(noutputs,1)),
                               np.reshape(np.ones(noutputs)*ps1, (noutputs,1)),
                               np.reshape(np.ones(noutputs)*m1, (noutputs,1)),
                               np.reshape(np.ones(noutputs)*s1, (noutputs,1)),
                               p,
                               vp,
                               np.reshape(np.ones(noutputs)*ps2, (noutputs,1)),
                               np.reshape(np.ones(noutputs)*m2, (noutputs,1)),
                               np.reshape(np.ones(noutputs)*s2, (noutputs,1)),
                               s,
                               vs,
                               np.reshape(np.ones(noutputs)*ps3, (noutputs,1)),
                               np.reshape(np.ones(noutputs)*mimp,(noutputs,1)),
                               np.reshape(np.ones(noutputs)*simp[i],(noutputs,1)),
                               imp,
                               vimp))
        
        df = pd.DataFrame(particles)
        
        # df.to_csv(f'./results/{sim_name}_{np.round(simp[i]/1e3, 1)}_{np.round(b[j], 1)}.csv', header=headers)
        df.to_csv(f'./results/{sim_name}_{starting_position}.csv', header=headers)
            
print(timed()-timer) # finish timer