import numpy as np

def rndm(a, b, g, size=1):
    r = np.random.random(size=size)
    ag, bg = a**g, b**g
    return (ag + (bg - ag)*r)**(1./g)