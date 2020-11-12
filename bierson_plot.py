#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 15:56:32 2020

@author: john
"""

import numpy as np
import matplotlib.pyplot as plt

densities = np.array([600,1370,600,1140,300,730,500,1040,820,1270,1260,1520,2180,1700,1885,2520,1850,2060])
radii = np.array([157,174,178,235,123,249,304,632,652,705,866,958,1070,1212,1595,2326,2376,2706])/2

plt.scatter(radii, densities)
plt.xlabel("radius [km]")
plt.ylabel(r"density [kg/m${^3}$]")
plt.ylim(0)
# plt.savefig("bierson.png")

low_densities = densities[radii < 200]
average_density = np.mean(low_densities)