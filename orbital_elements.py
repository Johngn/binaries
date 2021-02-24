import numpy as np

G = 6.67428e-11
mu = G*1e5

t = 0

def kep_2_cart(a,e,omega,f,EA):
    
    n = np.sqrt(mu/(a**3))
    M = n*(t - f)
    
    MA = EA - e*np.sin(EA)
    
    nu = 2*np.arctan(np.sqrt((1+e)/(1-e)) * np.tan(EA/2))
    
    r = a*(1 - e*np.cos(EA))
    
    h = np.sqrt(mu*a * (1 - e**2))

    X = r*(np.cos(omega+nu))
    Y = r*(np.sin(omega+nu))
    
    p = a*(1-e**2)

    V_X = (X*h*e/(r*p))*np.sin(nu) - (h/r)*(np.sin(omega+nu))
    V_Y = (Y*h*e/(r*p))*np.sin(nu) - np.cos(omega+nu)

    return [X,Y],[V_X,V_Y]

a = 1e3
e = 0
omega = 0
f = 0
EA = 0

r_test2, v_test2 = kep_2_cart(a,e,omega,f,EA)

print(r_test2)
print(v_test2)