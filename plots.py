# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

G = 6.67428e-11
au = 1.496e11
rsun = 44.*au
Msun = 1.9891e30

sim_name = 'OCT11_dmr'

data = pd.read_csv(f'./results/{sim_name}_final.csv')

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

R, V, mu, h = np.zeros((len(data),3)), np.zeros((len(data),3)), np.zeros((len(data),3)), np.zeros((len(data),3))
R[:,0] = np.linalg.norm(p-s, axis=1)
R[:,1] = np.linalg.norm(p-imp, axis=1)
R[:,2] = np.linalg.norm(s-imp, axis=1)
V[:,0] = np.linalg.norm(vp-vs, axis=1)
V[:,1] = np.linalg.norm(vp-vimp, axis=1)
V[:,2] = np.linalg.norm(vs-vimp, axis=1)
h[:,0] = np.cross(p-s,vp-vs)[:,2]
h[:,1] = np.cross(p-imp,vp-vimp)[:,2]
h[:,2] = np.cross(s-imp,vs-vimp)[:,2]
mu[:,0] = G*(m1+m2)
mu[:,1] = G*(m1+mimp)
mu[:,2] = G*(m2+mimp)

Rhill = np.array([rsun*(m1/Msun/3.)**(1./3.), rsun*(m2/Msun/3.)**(1./3.), rsun*(mimp/Msun/3.)**(1./3.)])
Rhill_largest = np.array([np.amax([Rhill[0], Rhill[1]]), np.amax([Rhill[0], Rhill[2]]), np.amax([Rhill[1], Rhill[2]])])

a = mu*R/(2*mu - R*V**2)
energy = -mu/2/a
e = np.sqrt(1 + (2*energy*h**2 / mu**2))

bound = np.logical_and(np.logical_and(energy < 0, np.isfinite(energy)), R < Rhill_largest)
collision = R[:,0] == 0

plt.figure(figsize=(6,6))
s = 40
plt.scatter(b, simp, s=1, marker="x", c="black")
plt.scatter(b[bound[:,0]], simp[bound[:,0]], label='primary-secondary', s=s)
plt.scatter(b[bound[:,1]], simp[bound[:,1]], label='primary-impactor', s=s)
plt.scatter(b[bound[:,2]], simp[bound[:,2]], label='secondary-impactor', s=s)
plt.scatter(b[collision], simp[collision], label='collision', s=s)
plt.xlabel("Impact parameter (Hill radii)")
plt.ylabel("Impactor radius (km)")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
# plt.xticks(np.arange(0.5,10.6,0.5))
# plt.yticks(np.arange(0,101,5))
# plt.savefig(f"./img/final_bound", bbox_inches='tight')
# %%
binary_e = np.reshape(e[:,0], (len(np.unique(simp)), len(np.unique(b))))
plt.figure(figsize=(8,6))
ax = sns.heatmap(binary_e, 
                 annot=True, 
                 linewidths=0.5, 
                 cmap="YlGnBu",
                 yticklabels=np.asarray((np.flip(np.unique(simp))), dtype=int),
                 xticklabels=np.round(np.unique(b), 2))
plt.xlabel("Impact parameter (Hill radii)")
plt.ylabel("Impactor radius (km)")