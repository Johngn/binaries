import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

densities =         np.array([600,1370,600,1140,300,730,500,1040,820,1270,1260,1520,2180,1700,1885,2520,1850,2060])
density_errors =    np.array([200, 320,400, 300,150,300,100, 170,100, 400, 300, 150, 400,  10,  80,  50,   6,   0])

diameter = np.array([157,174,178,235,123,249,304,632,652,705,866,958,1070,1212,1595,2326,2376,2706])
diameter_errors = np.array([34,17,35,22,40,30,101,34,12,79,37,23,38,2,11,12,3,1.8])

fig, ax = plt.subplots(1, figsize=(5,4))

# plt.scatter(radii, densities, color="black", edgecolors="", s=20)
plt.errorbar(diameter, densities, yerr=density_errors, xerr=diameter_errors, fmt='o', markersize=3, elinewidth=1, color="teal")
plt.xlabel("Diameter [km]")
plt.ylabel(r"Density [kg/m${^3}$]")
plt.ylim(0)

plt.savefig("img/bierson.pdf", bbox_inches='tight')

# low_densities = densities[radii < 200]
# average_density = np.mean(low_densities)