import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from powerlaw import rndm

fig, ax = plt.subplots(1, 1)

a = [rndm(1, 300, g=-1.6, size=1)[0]*1e3 for i in range(100000)]

# sns.distplot(a, bins=1000, kde=True, norm_hist=False,
               # kde_kws={"color": "darkred", "lw": 1})
sns.distplot(a, bins=2000, kde=True, hist=False,
             # kde_kws={"color": "darkred", "lw": 1}
             )

ax.set_xlim(0)
