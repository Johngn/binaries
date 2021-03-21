import numpy as np

m_tot = 1.2e23
m_max = 2e20
m_min = 0

exp_1 = 0.4

k = m_tot * exp_1 / ( m_max**(exp_1) - m_min**(exp_1) )

exp_2 = -0.6
m_max = 3e18
m_min = 3e15

N = k / exp_2 * ( m_max**(exp_2) - m_min**(exp_2) )

# %%
import numpy as np

au = 1.496e11
g = 6.67428e-11
msun = 1.9891e30
Omega_k = np.sqrt(g*msun/(44*au)**3)

N = 1e6

h = 8*au
V = h*np.pi*((47*au)**2 - (42*au)**2)
rho = N/V

year = 3.16e7
T = 4e9*year

rsun = 44*au
dens = 700
s1 = 100e3
m1 = 4./3.*np.pi*dens*s1**3
rhill = rsun*(m1/msun/3)**(1/3)
r = 5*rhill
sigma = np.pi*r**2

v_shear = -1.5*Omega_k*r

dv = 2
C = rho*dv*sigma
n_enc = C*T
