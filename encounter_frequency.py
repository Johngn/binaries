import numpy as np


m_tot = 6e22
m_max = 3e15
m_min = 3e1

exp_1 = 0.4

x = m_tot * exp_1 / ( m_max**(exp_1) - m_min**(exp_1) )

exp_2 = -0.6
m_max = 3e15
m_min = 3e9

N = x / exp_2 * ( m_max**(exp_2) - m_min**(exp_2) )

# %%
N = np.arange(1e4,1e9,1e5)
N = 2e7

au = 1.496e11

V = np.pi*((47*au)**2 - (42*au)**2)*8*au
rho = N/V

year = 3.16e7
T = 4.5e9*year

msun = 1.9891e30
rsun = 44.*au
dens = 700.
s1 = 100e3
m1 = 4./3.*np.pi*dens*s1**3
rhill = rsun*(m1/msun/3.)**(1./3.)
r = 8*rhill
sigma = np.pi*r**2

dv = np.arange(0,1000,1)

C = rho*dv*sigma
n_enc = C*T

# %%

import matplotlib.pyplot as plt

plt.plot(dv, C)