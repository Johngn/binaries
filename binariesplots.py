# %%
import glob, os, csv, rebound, mysql.connector, pymysql
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation
from timeit import default_timer as timed
from sqlalchemy import create_engine

G = 6.67428e-11                             # gravitational constanct in SI units
au = 1.496e11                               # astronomical unit    
Msun = 1.9891e30                            # mass of sun
rsun = 44.*au                               # distance of centre of mass of binary from the sun 
T = 2.*np.pi/np.sqrt(G*(Msun)/rsun**3)      # orbital period of binary around the sun
n = 2*np.pi/T                               # mean motion of binary around the sun
year = 365.25*24.*60.*60.                   # number of seconds in a year
Noutputs = 1000                             # number of outputs for plotting

# db_connection_str = 'mysql+pymysql://john:321654@localhost/mydatabase'
# db_connection = create_engine(db_connection_str)
# mydb = mysql.connector.connect(
#   host="localhost",
#   user="john",
#   passwd="321654",
#   database="mydatabase"
# )
# mycursor = mydb.cursor()
# mycursor.execute("SHOW TABLES")
# tablenames = np.array(mycursor.fetchall())[:,0]
# results = [pd.read_sql_table(i, con=db_connection) for i in tablenames]

# filenames = glob.glob(f"{path}/results/particles*.csv")
# results = [pd.read_csv(i, delimiter=',') for i in filenames]

# read data from csv into dataframe
data = pd.read_csv(f'./results/particles__b-5.0__r-30.0.csv')

# create numpy arrays from dataframe
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

# empty arrays for values
r, v, Rhill, mu, h, e = np.zeros((len(data),3)), np.zeros((len(data),3)), np.zeros((len(data),3)), np.zeros((len(data),3)),np.zeros((len(data),3)),np.zeros((len(data),3))

r[:,0] = np.linalg.norm(p-s, axis=1)                # distance between primary and secondary
r[:,1] = np.linalg.norm(p-imp, axis=1)              # distance between primary and impactor
r[:,2] = np.linalg.norm(s-imp, axis=1)              # distance between secondary and impactor

v[:,0] = np.linalg.norm(vp-vs, axis=1)              # relative velocity between primary and secondary
v[:,1] = np.linalg.norm(vp-vimp, axis=1)            # relative velocity between primary and impactor
v[:,2] = np.linalg.norm(vs-vimp, axis=1)            # relative velocity between secondary and impactor

Rhill[:,0] = rsun*((m1+m2)/Msun/3.)**(1./3.)        # combined Hill radius of primary and secondary
Rhill[:,1] = rsun*((m1+mimp)/Msun/3.)**(1./3.)      # combined Hill radius of primary and impactor
Rhill[:,2] = rsun*((m2+mimp)/Msun/3.)**(1./3.)      # combined Hill radius of secondary and impactor

mu[:,0] = G*(m1+m2)                                 # G times combined mass of primary and secondary
mu[:,1] = G*(m1+mimp)                               # G times combined mass of primary and impactor
mu[:,2] = G*(m2+mimp)                               # G times combined mass of secondary and impactor

a = mu*r/(2*mu - r*v**2)                            # semi-major axis between each pair of bodies
energy = -mu/2/a                                    # total energy between each pair of bodies
bound = np.logical_and(energy < 0, r < Rhill)       # bodies are bound if their energy is less than zero and they are closer together than the Hill radius

distance1 = p-s                                     # difference between x, y and z values of primary and secondary
distance2 = p-imp                                # difference between x, y and z values of primary and secondary
distance3 = s-imp
v1 = vp-vs
v2 = vp-vimp
v3 = vs-vimp
h[:,0] = np.cross(distance1,v1)[:,2]
h[:,1] = np.cross(distance2,v2)[:,2]
h[:,2] = np.cross(distance3,v3)[:,2]
e = np.sqrt(1 + (2 * energy * h**2 / mu**2))

OmegaK = np.sqrt(G*(Msun+m1[0]+m2[0])/rsun**3)
angles = -OmegaK*times

sp = (s-p)/Rhill[0,0]
ip = (imp-p)/Rhill[0,1]
cosspx, cosspy = np.cos(angles)*sp[:,0], np.cos(angles)*sp[:,1]
sinspx, sinspy = np.sin(angles)*sp[:,0], np.sin(angles)*sp[:,1]

xdot = vs[:,0] - vp[:,0]
ydot = vs[:,1] - vp[:,1]
cosspxdot, cosspydot = np.cos(angles)*xdot, np.cos(angles)*ydot
sinspxdot, sinspydot = np.sin(angles)*xdot, np.sin(angles)*ydot

x, y = cosspx-sinspy+rsun, sinspx+cosspy
vx, vy = cosspxdot-sinspydot, sinspxdot+cosspydot

Cj = n**2*(x**2 + y**2) + 2*(mu[:,0]/r[:,0] + mu[:,1]/r[:,1]) - vx**2 - vy**2
# %%
lim = 20
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
primaryhill = plt.Circle((0,0), Rhillprim[0], fc="none", ec="tab:orange")
secondaryhill = plt.Circle((0,0), Rhillsec[0], fc="none", ec="tab:blue")
impactorhill = plt.Circle((0,0), Rhillimp[0], fc="none", ec="tab:green")

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
    text.set_text('{} Years'.format(np.round(times[i]/(year), 1)))
    return primarydot, secondarydot, impactordot, primaryline, secondaryline, impactorline, text, primaryhill, secondaryhill, impactorhill

anim = animation.FuncAnimation(fig, animate, init_func=init,  frames=Noutputs, interval=1,blit=True)
# anim.save(f'{path}/videos/2D.mp4')
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
    text.set_text('{} Years'.format(np.round(times[i]/(year), 1)))
    return primarydot, secondarydot, impactordot, primaryline, secondaryline, impactorline, text

anim = animation.FuncAnimation(fig, animate, blit=True, frames=len(data), interval=1)
# anim.save(f'{path}/videos/3D.mp4')
# %%
