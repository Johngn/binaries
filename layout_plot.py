import numpy as np
import matplotlib.pyplot as plt

theta = np.linspace(0, 2*np.pi, 100)

r1 = 20.0
r2 = 25.0

x1 = r1*np.cos(theta)
y1 = r1*np.sin(theta)

x2 = r2*np.cos(theta)
y2 = r2*np.sin(theta)

fig, ax = plt.subplots(1, figsize=(8,8))
plt.axis('off')
ax.plot(x1, y1, lw=1, color='tab:orange')
ax.plot(x2, y2, lw=1, color='tab:green')
ax.scatter(0,0, color='gold', s=200)
ax.scatter(19.5, 0, label='Primary', color='tab:orange', s=100)
ax.scatter(20.5, 0, label='Secondary', color='tab:blue', s=100)
ax.scatter(24.3, 6, label='Impactor', color='tab:green', s=100)
ax.legend(loc='upper right')

ax.set_aspect(1)

fig.savefig('layout.pdf', bbox_inches='tight')