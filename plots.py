import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import matplotlib as mpl

G = 6.67428e-11
au = 1.496e11
rsun = 44.*au
Msun = 1.9891e30

sim_name = 'verytight_equalmass_ecc0_0_'
filenames = glob(f'./thesis_results/{sim_name}*')

b_all = np.zeros(len(filenames))
simp_all = np.zeros(len(filenames))
e_mean = np.zeros(len(filenames))
e_final = np.zeros((len(filenames),3))
a_all = np.zeros((len(filenames),3))
bound = np.zeros((len(filenames), 3))

for i, sim in enumerate(filenames):
    
    data = np.array(pd.read_csv(sim, index_col=0))

    times = data[:,0]
    b = data[:,1]
    hash_primary = data[:,2]
    m1 = data[0,3]
    p = data[:,5:8]
    vp = data[:,8:11]
    hash_secondary = data[:,11]
    m2 = data[0,12]
    s = data[:,14:17]
    vs = data[:,17:20]
    hash_impactor = data[:,20]
    mimp = data[0,21]
    simp = data[:,22]
    imp = data[:,23:26]
    vimp = data[:,26:29]
    
    b_all[i] = b[-1]
    simp_all[i] = simp[-1]

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
    e_movingavg = [np.mean(e[ii-100:ii,:], axis=0) for ii in range(len(data))]
    
    a_all[i] = a[-1]
    
    bound[i] = np.logical_and(np.logical_and(energy < 0, np.isfinite(energy)), R < Rhill_largest)[-1]
    
    e_final[i] = e_movingavg[-1]
    

a_all = a_all/Rhill[0]
b_all = b_all*Rhill[0]
simp_all = simp_all/1e3

bound = np.array(bound, dtype='bool')

collisions = glob(f'./thesis_results/collision_{sim_name}*')
coll_params = np.zeros((len(collisions), 3))
for i, collision in enumerate(collisions):
    params = re.findall('[0-9]+\.[0-9]+', collision)
    params = [float(param) for param in params]
    
    collided_bodies = np.array(pd.read_csv(collision, index_col=0))[:,1]
    
    # if hash_primary[0] in collided_bodies and hash_secondary[0] in collided_bodies:
    if np.array_equal(collided_bodies, [0,1]):
        params.append(1)
    # if hash_primary[0] in collided_bodies and hash_impactor[0] in collided_bodies:
    if np.array_equal(collided_bodies, [0,3]):
        params.append(2)
    # if hash_secondary[0] in collided_bodies and hash_impactor[0] in collided_bodies:
    if np.array_equal(collided_bodies, [1,3]):
        params.append(3)
    if len(params) == 3: # needed if more than one collision per sim
        coll_params[i] = params
    
# %%
fig, ax = plt.subplots(1, figsize=(4,5))
s = 65
c1 = "#444e86"
c2 = "#c34f8e"
c3 = "#ff6e54"
c4 = ''
# ax.scatter(b_all, simp_all, s=1, marker="o", c="black")
ax.scatter(b_all[bound[:,0]], simp_all[bound[:,0]], label='Prim-Sec', s=s, c=c1, marker='s', edgecolor=c4)
ax.scatter(b_all[bound[:,1]], simp_all[bound[:,1]], label='Prim-Imp', s=s, c=c2, marker='s', edgecolor=c4)
ax.scatter(b_all[bound[:,2]], simp_all[bound[:,2]], label='Sec-Imp', s=s, c=c3, marker='s', edgecolor=c4)
ax.scatter(coll_params[coll_params[:,2] == 1][:,1], coll_params[coll_params[:,2] == 1][:,0], marker='x', s=s*0.7, c=c1, edgecolor=c4)
ax.scatter(coll_params[coll_params[:,2] == 2][:,1], coll_params[coll_params[:,2] == 2][:,0], marker='x', s=s*0.7, c=c2, edgecolor=c4)
ax.scatter(coll_params[coll_params[:,2] == 3][:,1], coll_params[coll_params[:,2] == 3][:,0], marker='x', s=s*0.7, c=c3, edgecolor=c4)
ax.set_xlabel("Impact parameter [$r_{\mathrm{H}, prim}$]")
ax.set_ylabel("Impactor radius [km]")
# ax.set_title(f'initial semi-major axis = 0.4 R$_H$', y=1, pad=15, fontdict={'fontsize': 14})
ax.legend(loc='upper center', bbox_to_anchor=(0.49, 1.06), ncol=3, fancybox=True, shadow=True)
# ax.set_yticks(np.asarray((np.unique(simp_all)), dtype=int),)
# ax.set_xticks(np.round(np.unique(b_all), 2))
# plt.xlim(100,350)simp
plt.savefig(f"./img/{sim_name}_final_bound.pdf", bbox_inches='tight')


# %%
test0 = e_final[:,0]
test1 = e_final[:,1]
test2 = e_final[:,2]
test0[e_final[:,0] > 1] = 0
test1[e_final[:,1] > 1] = 0
test2[e_final[:,2] > 1] = 0
test = test0+test1+test2
s = 45
fig, ax = plt.subplots(1, figsize=(4,4))
cm = plt.cm.get_cmap('YlGnBu')

ax.scatter(b_all[bound[:,0]], simp_all[bound[:,0]], label='Prim-Sec', s=s, vmin=0, vmax=1,
           c=test[bound[:,0]], marker='s', edgecolor=c4, cmap=cm)
ax.scatter(b_all[bound[:,1]], simp_all[bound[:,1]], label='Prim-Imp', s=s*0.5, vmin=0, vmax=1,
           c=test[bound[:,1]], marker='D', edgecolor=c4, cmap=cm)
ax.scatter(b_all[bound[:,2]], simp_all[bound[:,2]], label='Sec-Imp', s=s*0.5, vmin=0, vmax=1,   
           c=test[bound[:,2]], marker='D', edgecolor=c4, cmap=cm)
ax.set_xlabel("Impact parameter [$r_{\mathrm{H}, prim}$]")
ax.set_ylabel("Impactor radius [km]")
# plt.colorbar(sc)
norm = mpl.colors.Normalize(vmin=0, vmax=1)
plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cm), label='Eccentricity')

plt.savefig(f"./img/{sim_name}_final_eccentricity.pdf", bbox_inches='tight')
# %%

b_all = np.round(b_all, 2)
binary_e = np.ones((len(np.unique(simp_all)), len(np.unique(b_all))))

for i, item in enumerate(np.unique(simp_all)):
    for j, jtem in enumerate(np.unique(b_all)):
        ecc = test[np.logical_and(np.round(simp_all,2) == item, b_all == jtem)]
        if len(ecc > 0):
            binary_e[i,j] = ecc
            
binary_e[binary_e > 1] = np.nan
binary_e[binary_e == 0] = np.nan
binary_e = np.round(binary_e, 2)

fig, ax = plt.subplots(1, figsize=(4,4))
ax = sns.heatmap(binary_e, 
                 # annot=True,
                 linewidths=0.5, 
                 cmap="YlGnBu",
                  # yticklabels=np.asarray((np.unique(simp_all)), dtype=int),
                 yticklabels=[50,'','','','',100,'','','','',150,'','','','',200,'','','','',250,'','','','',300,],     
                 # xticklabels=np.round(np.unique(b_all), 2),
                  # yticklabels=np.asarray((np.unique(simp_all)), dtype=int),
                 xticklabels=[2,'','','','',3,'','','','',4,'','','','',5,'','','','',6],
                 # cbar=False
                 cbar_kws={'label': 'Eccentricity'}
                 )
ax.invert_yaxis()
plt.ylabel("Impactor radius [km]")
plt.xlabel("Impact parameter [$r_{\mathrm{H}, prim}$]")
plt.savefig(f"./img/{sim_name}_final_eccentricity.pdf", bbox_inches='tight')

# %%
test0 = a_all[:,0]
test1 = a_all[:,1]
test2 = a_all[:,2]
test0[a_all[:,0] > 1] = 0
test1[a_all[:,1] > 1] = 0
test2[a_all[:,2] > 1] = 0
test0[a_all[:,0] < 0] = 0
test1[a_all[:,1] < 0] = 0
test2[a_all[:,2] < 0] = 0
test = test0+test1+test2
c4 = ''
fig, ax = plt.subplots(1, figsize=(4,4))
cm = plt.cm.get_cmap('YlGnBu')

ax.scatter(b_all[bound[:,0]], simp_all[bound[:,0]], label='Prim-Sec', s=s, vmin=0, vmax=1,
           c=test[bound[:,0]], marker='s', edgecolor=c4, cmap=cm)
ax.scatter(b_all[bound[:,1]], simp_all[bound[:,1]], label='Prim-Imp', s=s*0.5, vmin=0, vmax=1,
           c=test[bound[:,1]], marker='D', edgecolor=c4, cmap=cm)
ax.scatter(b_all[bound[:,2]], simp_all[bound[:,2]], label='Sec-Imp', s=s*0.5, vmin=0, vmax=1,   
           c=test[bound[:,2]], marker='D', edgecolor=c4, cmap=cm)
ax.set_xlabel("Impact parameter [$r_{\mathrm{H}, prim}$]")
ax.set_ylabel("Impactor radius [km]")
# plt.colorbar(sc)
norm = mpl.colors.Normalize(vmin=0, vmax=1)
plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cm), label='Mutual orbital separation [$r_{\mathrm{H}}$]')

plt.savefig(f"./img/{sim_name}_final_semimajoraxis.pdf", bbox_inches='tight')
# %%
binary_a = np.ones((len(np.unique(simp_all)), len(np.unique(b_all))))

for i, item in enumerate(np.unique(simp_all)):
    for j, jtem in enumerate(np.unique(b_all)):
        sma = test[np.logical_and(np.round(simp_all,2) == item, b_all == jtem)]
        if len(sma > 0):
            binary_a[i,j] = sma
            
binary_a = np.round(binary_a, 2)
binary_a[binary_a <= 0] = np.nan
binary_a[binary_a > 1] = np.nan

fig, ax = plt.subplots(1, figsize=(4, 4))
ax = sns.heatmap(binary_a, 
                 linewidths=0.5, 
                 cmap="YlGnBu",
                  # yticklabels=np.asarray((np.unique(simp_all)), dtype=int),
                 yticklabels=[50,'','','','',100,'','','','',150,'','','','',200,'','','','',250,'','','','',300,],     
                 # xticklabels=np.round(np.unique(b_all), 2),
                  # yticklabels=np.asarray((np.unique(simp_all)), dtype=int),
                 xticklabels=[2,'','','','',3,'','','','',4,'','','','',5,'','','','',6],
                 # cbar=False
                 cbar_kws={'label': 'a/$r_{\mathrm{H}}$'}
                 )
ax.invert_yaxis()
# plt.title(f"semi-major axis ({sim_name})")
plt.xlabel("Impact parameter [$r_{\mathrm{H}, prim}$]")
plt.ylabel("Impactor radius [km]")
plt.savefig(f"./img/{sim_name}_final_semimajoraxis.pdf", bbox_inches='tight')
# %%
binary_a = np.ones((len(np.unique(simp)), len(np.unique(b))))

for i, item in enumerate(np.unique(simp)):
    for j, jtem in enumerate(np.unique(b)):
        sma = periapsis[np.logical_and(np.round(simp,2) == item, b == jtem)][:,0]
        if len(sma > 0):
            binary_a[i,j] = sma
            
binary_a = np.round(binary_a/Rhill[0], 2)
binary_a[binary_a <= 0] = np.nan
binary_a[binary_a > 10] = np.nan

fig, ax = plt.subplots(1, figsize=(12, 9))
ax = sns.heatmap(binary_a, 
                 annot=True, 
                 linewidths=0.5, 
                 cmap="YlGnBu",
                 yticklabels=np.asarray((np.unique(simp)), dtype=int),
                 xticklabels=np.round(np.unique(b), 2),
                 cbar=False)
ax.invert_yaxis()
plt.title(f"periapsis ({sim_name})")
plt.xlabel("Impact parameter (Hill radii)")
plt.ylabel("Impactor radius (km)")
# plt.savefig(f"./img/{sim_name}_final_periapsis", bbox_inches='tight')
# %%

data = {'bound':  number_of_bound,
        'swapped': number_of_swapped,
        'collided': number_of_collisions,
        'disrupted': number_of_disrupted
        }

df = pd.DataFrame(data)
# plt.bar(number_of_bound, number_of_swapped)
# %%
total = len(filenames)
number_of_bound = np.count_nonzero(bound[:,0])/total
number_of_swapped1 = np.count_nonzero(bound[:,1])/total
number_of_swapped2 = np.count_nonzero(bound[:,2])/total
number_of_collisions = len(collisions)
disrupted = [not bound[i].any() for i in range(len(bound))]
number_of_disrupted = np.count_nonzero(disrupted)/total

fig, ax = plt.subplots(1, figsize=(5,4))
ax.bar([1,2,3,4], [number_of_bound,number_of_swapped1,number_of_swapped2,number_of_disrupted] , width=1, color=['#444e86','#c34f8e','#ff6e54','#ffa600'], edgecolor="black")

# ax.set_ylabel('Number of encounter')

ax.grid(color='black', linestyle='--', linewidth=1, axis='y', alpha=0.2)

plt.xticks([1,2,3,4], ('Prim-Sec', 'Prim-Imp', 'Sec-Imp', 'Disrupted'))
plt.savefig(f"./img/{sim_name}_dist.pdf", bbox_inches='tight')