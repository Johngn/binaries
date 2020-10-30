# %%
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

G = 6.67428e-11
au = 1.496e11
rsun = 44.*au
Msun = 1.9891e30

sim_name = 'OCT26-2'

# data = pd.read_csv(f'./results/{sim_name}_final.csv')
data = glob(f'./results/{sim_name}*')

final = np.zeros((len(data), 25))
for i, sim in enumerate(data):
    final[i] = np.array(pd.read_csv(sim))[-1]

collisions = glob(f'./results/collision_{sim_name}*')
coll_params = np.zeros((len(collisions), 3))
for i, collision in enumerate(collisions):
    params = re.findall('[0-9]+\.[0-9]+', collision)
    params = [float(param) for param in params]
    
    collided_bodies = pd.read_csv(collision)['body'].values
    if collided_bodies[0] == 1 and collided_bodies[1] == 2:
        params.append(1)
    if collided_bodies[0] == 1 and collided_bodies[1] == 3:
        params.append(2)
    if collided_bodies[0] == 2 and collided_bodies[1] == 3:
        params.append(3)
    coll_params[i] = params

b = final[:,2]
simp = final[:,3]
m1 = final[:,4]
p = final[:,5:8]
vp = final[:,8:11]
m2 = final[:,11]
s = final[:,12:15]
vs = final[:,15:18]
mimp = final[:,18]
imp = final[:,19:22]
vimp = final[:,22:25]

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
# collision = R[:,0] == 0
# %%
plt.figure(figsize=(10,9))
s = 100
plt.scatter(simp, b, s=1, marker="x", c="black")
plt.scatter(simp[bound[:,0]], b[bound[:,0]], label='primary-secondary', s=s, c="tab:blue")
plt.scatter(simp[bound[:,1]], b[bound[:,1]], label='primary-impactor', s=s, c="tab:orange")
plt.scatter(simp[bound[:,2]], b[bound[:,2]], label='secondary-impactor', s=s, c="tab:green")
# plt.scatter(simp[collision], b[collision], label='collision', s=s)
plt.scatter(coll_params[coll_params[:,2] == 1][:,1], coll_params[coll_params[:,2] == 1][:,0], marker='x', s=s, c="tab:blue")
plt.scatter(coll_params[coll_params[:,2] == 2][:,1], coll_params[coll_params[:,2] == 2][:,0], marker='x', s=s, c="tab:orange")
plt.scatter(coll_params[coll_params[:,2] == 3][:,1], coll_params[coll_params[:,2] == 3][:,0], marker='x', s=s, c="tab:green")
plt.ylabel("Impact parameter (Hill radii)")
plt.xlabel("Impactor radius (km)")
plt.title("2:1 mass ratio - wide separation")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
# plt.yticks(np.arange(0.5,10.6,0.5))
# plt.yticks(np.arange(0,101,5))
# plt.savefig(f"./img/final_bound_whr_dmr", bbox_inches='tight')
# %%
binary_e = np.zeros((len(np.unique(b)), len(np.unique(simp))))

for i, item in enumerate(np.unique(b)):
    for j, jtem in enumerate(np.unique(simp)):
        # row = final[np.logical_and(final[:,2] == item, final[:,3] == jtem)]
        ecc = e[np.logical_and(final[:,2] == item, final[:,3] == jtem)][:,0]
        binary_e[i,j] = ecc[0]
        # print(binary_e[i,j])
        
    # print(item)

# for i, row in enumerate(final):
#     a, b = row[2:4]
#     test = e[i,0]
    
    # print(row[2:4])

# %%
# binary_e = np.reshape(e[:,0], (len(np.unique(b)), -1))
binary_e[binary_e > 1] = np.nan
binary_e = np.round(binary_e, 2)

fig, ax = plt.subplots(1, figsize=(20, 10))
ax = sns.heatmap(binary_e, 
                 annot=True, 
                 linewidths=0.5, 
                 cmap="YlGnBu",
                 # xticklabels=np.asarray((np.unique(simp)), dtype=int),
                 # yticklabels=np.round(np.unique(b), 2)
                 )
# ax.invert_xaxis()
ax.invert_yaxis()
plt.title("Final eccentricity of binary")
plt.ylabel("Impact parameter (Hill radii)")
plt.xlabel("Impactor radius (km)")
# %%
binary_a = np.reshape(a[:,0], (len(np.unique(b)), len(np.unique(simp))))/Rhill[0,0]
binary_a[binary_a < 0] = np.nan
binary_a[binary_a > 1] = np.nan
binary_a = np.round(binary_a, 2)

fig, ax = plt.subplots(1, figsize=(18, 10))
ax = sns.heatmap(binary_a, 
                 annot=True, 
                 linewidths=0.5, 
                 cmap="YlGnBu",
                 xticklabels=np.asarray((np.unique(simp)), dtype=int),
                 yticklabels=np.round(np.unique(b), 2))
ax.invert_yaxis()
plt.title("Final semi-major axis of binary")
plt.ylabel("Impact parameter (Hill radii)")
plt.xlabel("Impactor radius (km)")