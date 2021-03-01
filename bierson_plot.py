import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

densities = np.array([600,1370,600,1140,300,730,500,1040,820,1270,1260,1520,2180,1700,1885,2520,1850,2060])
radii = np.array([157,174,178,235,123,249,304,632,652,705,866,958,1070,1212,1595,2326,2376,2706])/2

fig, ax = plt.subplots(1, figsize=(9,5))

plt.scatter(radii, densities, color="steelblue", edgecolors="black", s=80)
plt.xlabel("radius [km]")
plt.ylabel(r"density [kg/m${^3}$]")
plt.ylim(0)

plt.savefig("img/bierson.pdf")

low_densities = densities[radii < 200]
average_density = np.mean(low_densities)