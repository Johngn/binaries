# %%
import glob, os, csv, rebound, mysql.connector, pymysql
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation
from timeit import default_timer as timed
from sqlalchemy import create_engine

path = '/home/john/Desktop/mastersproject'
G = 6.67428e-11
au = 1.496e11
rsun = 44.*au
Msun = 1.9891e30
T = 2.*np.pi/np.sqrt(G*(Msun)/rsun**3)
n = 2*np.pi/T
year = 365.25*24.*60.*60.
Noutputs = 1000

filenames = glob.glob(f"{path}/results/initial.csv")
results = [pd.read_csv(i, delimiter=',') for i in filenames]
data = pd.concat(results)

times = data['time'].to_numpy()
b = data['b'].to_numpy()
simp = data['imp radius'].to_numpy()
m1 = data['mass prim'].to_numpy()
m2 = data['mass sec'].to_numpy()
mimp = data['mass imp'].to_numpy()
p = data[['x prim','y prim', 'z prim']].to_numpy()
s = data[['x sec','y sec', 'z sec']].to_numpy()
imp = data[['x imp','y imp', 'z imp']].to_numpy()
vp = data[['vx prim','vy prim', 'vz prim']].to_numpy()
vs = data[['vx sec','vy sec', 'vz sec']].to_numpy()
vimp = data[['vx imp','vy imp', 'vz imp']].to_numpy()

r, v, Rhill, mu = np.zeros((len(data),3)), np.zeros((len(data),3)), np.zeros((len(data),3)), np.zeros((len(data),3))
r[:,0] = np.linalg.norm(p-s, axis=1)
r[:,1] = np.linalg.norm(p-imp, axis=1)
r[:,2] = np.linalg.norm(s-imp, axis=1)
v[:,0] = np.linalg.norm(vp-vs, axis=1)
v[:,1] = np.linalg.norm(vp-vimp, axis=1)
v[:,2] = np.linalg.norm(vs-vimp, axis=1)
Rhill[:,0] = rsun*((m1+m2)/Msun/3.)**(1./3.)
Rhill[:,1] = rsun*((m1+mimp)/Msun/3.)**(1./3.)
Rhill[:,2] = rsun*((m2+mimp)/Msun/3.)**(1./3.)
mu[:,0] = G*(m1+m2)
mu[:,1] = G*(m1+mimp)
mu[:,2] = G*(m2+mimp)

a = mu*r/(2*mu - r*v**2)
energy = -mu/2/a
bound = np.logical_and(energy < 0, r < Rhill)
# %%
plt.figure(figsize=(8,8))
plt.scatter(b[bound[:,0]],simp[bound[:,0]], label='primary-secondary', s=50)
plt.scatter(b[bound[:,1]],simp[bound[:,1]], label='primary-impactor', s=50)
plt.scatter(b[bound[:,2]],simp[bound[:,2]], label='secondary-impactor', s=50)
plt.legend()
plt.grid()
# %%
y = a
plt.figure(figsize=(15,8))
# plt.title(f"Integrator={sim.integrator} -- Impactor radius={simp/1e3} km -- b={B/Rhill} hill radii")
plt.plot(y[:,0], label="Primary-Secondary", lw=1.5)
plt.plot(y[:,1], label="Primary-Impactor", lw=1.5)
plt.plot(y[:,2], label="Secondary-Impactor", lw=1.5)
plt.axhline(y=0, ls="--", color="black", lw=1.5)
plt.xlabel("Impactor mass (years)")
plt.ylabel("Energy (J/kg)")
# plt.xlim(0, np.amax(times)/year)
# plt.ylim(-0.02, 0.02)
plt.grid('both')
plt.legend()
# plt.savefig(f"energy_{str(sim.integrator)}", bbox_inches='tight')