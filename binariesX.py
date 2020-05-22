# %%
import csv, rebound, mysql.connector, pymysql
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation
from scipy.spatial.distance import pdist
from timeit import default_timer as timed
from sqlalchemy import create_engine

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
rbin = 0.5*Rhill
vorb = np.sqrt(G*(m1+m2)/rbin)
vshear = -1.5*OmegaK*rbin
Pbin = 2.*np.pi/np.sqrt(G*(m1+m2)/rbin**3)
T = 2.*np.pi/np.sqrt(G*(Msun)/r**3)
n = 2*np.pi/T
y0 = 10*Rhill
mu1 = G*Msun
mu2 = G*m1
path = '/home/john/Desktop/mastersproject/results'

simp = np.arange(10e3,50e3,10e3)
b = np.arange(4.0,5.5,0.5)*Rhill

binaryi = np.deg2rad(0)
impi = np.deg2rad(0)

Noutputs = 1000
totaltime = T
p, s, imp, sun = np.zeros((Noutputs, 3)), np.zeros((Noutputs, 3)), np.zeros((Noutputs, 3)), np.zeros((Noutputs, 3))
vp, vs, vimp, vsun = np.zeros((Noutputs, 3)), np.zeros((Noutputs, 3)), np.zeros((Noutputs, 3)), np.zeros((Noutputs, 3))
times = np.reshape(np.linspace(0.,totaltime, Noutputs), (Noutputs,1))
m1s = np.reshape(np.ones(Noutputs)*m1, (Noutputs,1))
m2s = np.reshape(np.ones(Noutputs)*m2, (Noutputs,1))
Msuns = np.reshape(np.ones(Noutputs)*Msun, (Noutputs,1))
headers = ['time','mass prim','x prim','y prim','z prim','vx prim','vy prim','vz prim',
           'mass sec','x sec','y sec','z sec','vx sec','vy sec','vz sec',
           'mass imp','x imp','y imp','z imp','vx imp','vy imp','vz imp',
           'mass sun','x sun','y sun','z sun','vx sun','vy sun','vz sun',]
db_connection_str = 'mysql+pymysql://john:321654@localhost/mydatabase'
db_connection = create_engine(db_connection_str)

# %%
k = 1
timer = timed()
for j in range(len(b)):
    for i in range(len(simp)):
        k += 1
        table_name = f'{str(int(simp[i]))}_{str(int(b[j]/Rhill))}'
        # table_name = f'{str(k)}'
        
        print(table_name)
        mimp = 4./3.*np.pi*densimp*simp[i]**3
        sim = rebound.Simulation()
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
        sim.add(m=m2, r=s2, x=xb2*np.sin(np.pi/2-binaryi), z=xb2*np.sin(binaryi), vy=vK2+vorb2*cosbin, vz=-vorb2*sinbin, hash="secondary")
        
        vorbi = np.sqrt(G*Msun/(r+b[j]))
        theta0 = y0/(r+b[j])
        stheta0 = np.sin(theta0)
        ctheta0 = np.cos(theta0)
        
        sim.add(m=mimp, r=simp[i], x=(r+b[j])*ctheta0-r*np.cos(impi), y=(r+b[j])*stheta0*np.cos(impi), 
                z=(r+b[j])*np.sin(impi), vx=-vorbi*stheta0, vy=vorbi*ctheta0, hash="impactor")

        ps = sim.particles        
        
        for k, time in enumerate(times):
            sim.integrate(time)    
            p[k] = [ps["primary"].x, ps["primary"].y, ps["primary"].z]
            s[k] = [ps["secondary"].x, ps["secondary"].y, ps["secondary"].z]
            imp[k] = [ps["impactor"].x, ps["impactor"].y, ps["impactor"].z]
            sun[k] = [ps["sun"].x, ps["sun"].y, ps["sun"].z]
            vp[k] = [ps["primary"].vx, ps["primary"].vy, ps["primary"].vz]
            vs[k] = [ps["secondary"].vx, ps["secondary"].vy, ps["secondary"].vz]
            vimp[k] = [ps["impactor"].vx, ps["impactor"].vy, ps["impactor"].vz]
            vsun[k] = [ps["sun"].vx, ps["sun"].vy, ps["sun"].vz]
            
        particles = np.hstack((times, m1s, p, vp, m2s, s, vs, np.reshape(np.ones(Noutputs)*mimp, (Noutputs,1)), imp, vimp, Msuns, sun, vsun))
        df = pd.DataFrame(particles)
        df.columns = headers
        # file = open(f'{path}/particles__b-{b[j]/Rhill}__r-{simp[i]/1e3}.csv', 'w')
        # with file:
        #     writer = csv.writer(file)
        #     writer.writerow(headers)
        #     writer.writerows(particles)
        # mycursor.execute(f"CREATE TABLE {table_name}")
        df.to_sql(f'{table_name}', con=db_connection, if_exists='replace')
            
print(timed()-timer)