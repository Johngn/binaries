#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 09:27:19 2020

@author: johngillan
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

G = 6.67428e-11
au = 1.496e11
rsun = 44.*au
Msun = 1.9891e30

filenames = glob(f'./rebound/mastersproject/binaries/results/*')
collision_filenames = glob(f'./rebound/mastersproject/binaries/results/collision*')
for i, filename in enumerate(filenames):
    if filename in collision_filenames:
        filenames.remove(filename)
        
results = []
        
for i, filename in enumerate(filenames):
    row = ["","","","","",""]
    
    if "verywide_" in filename:
        row[0] = "very wide"
    elif "wide_" in filename:
        row[0] = "wide"
    elif "verytight_" in filename:
        row[0] = "very tight"
    elif "tight_" in filename:
        row[0] = "tight"
    
    if "_equalmass_" in filename:
        row[1] = "equal mass"
    elif "_3mass_" in filename:
        row[1] = "3x mass"
    elif "_10mass_" in filename:
        row[1] = "10x mass"
    
    if "_0ecc_" in filename:
        row[2] = "0"
    elif "_1ecc_" in filename:
        row[2] = "0.1"
    elif "_2ecc_" in filename:
        row[2] = "0.2"
    elif "_3ecc_" in filename:
        row[2] = "0.3"
    elif "_4ecc_" in filename:
        row[2] = "0.4"
    elif "_5ecc_" in filename:
        row[2] = "0.5"
    elif "_6ecc_" in filename:
        row[2] = "0.6"
    elif "_7ecc_" in filename:
        row[2] = "0.7"
    elif "_8ecc_" in filename:
        row[2] = "0.8"
    elif "_9ecc_" in filename:
        row[2] = "0.9"
        
    row[3], row[4] = re.findall('[0-9]+\.[0-9]+', filename)
        
    cutoff_index = filename.rfind("/")
    collision_filename = "collision_" + filename[cutoff_index+1:]
            
    if "collision_" not in filename:
        data = np.loadtxt(filename)
        m1 = data[0,3]
        p = data[-1,5:8]
        vp = data[-1,8:11]
        m2 = data[0,12]
        s = data[-1,14:17]
        vs = data[-1,17:20]
        mimp = data[0,21]
        imp = data[-1,23:26]
        vimp = data[-1,26:29]
    
        R, V, mu, h = np.zeros((1,3)), np.zeros((1,3)), np.zeros((1,3)), np.zeros((1,3))
        R[:,0] = np.linalg.norm(p-s)
        R[:,1] = np.linalg.norm(p-imp)
        R[:,2] = np.linalg.norm(s-imp)
        V[:,0] = np.linalg.norm(vp-vs)
        V[:,1] = np.linalg.norm(vp-vimp)
        V[:,2] = np.linalg.norm(vs-vimp)
        h[:,0] = np.cross(p-s,vp-vs)[2]
        h[:,1] = np.cross(p-imp,vp-vimp)[2]
        h[:,2] = np.cross(s-imp,vs-vimp)[2]
        mu[:,0] = G*(m1+m2)
        mu[:,1] = G*(m1+mimp)
        mu[:,2] = G*(m2+mimp)
    
        Rhill = np.array([rsun*(m1/Msun/3.)**(1./3.), rsun*(m2/Msun/3.)**(1./3.), rsun*(mimp/Msun/3.)**(1./3.)])
        Rhill_largest = np.array([np.amax([Rhill[0], Rhill[1]]), np.amax([Rhill[0], Rhill[2]]), np.amax([Rhill[1], Rhill[2]])])
        
        a = mu*R/(2*mu - R*V**2)
        energy = -mu/2/a
        
        bound = np.logical_and(np.logical_and(energy < 0, np.isfinite(energy)), R < Rhill_largest)[-1]
        
        row[5] = "bound" if bound[0] else row[5]
        row[5] = "swapped" if bound[1] or bound[2] else row[5]
        row[5] = "disrupted" if not bound[0] and not bound[1] and not bound[2] else row[5]
        
    
    for j, collision in enumerate(collision_filenames):
        if collision_filename in collision:
            row[5] = "collision"
            # print("collision")
        
    # print(row[5])
        
    results.append(row)

df = pd.DataFrame(results, columns = ['a','mass ratio','eccentricity','impactor','b','status'])
df.to_csv('results.csv', index=False) 
