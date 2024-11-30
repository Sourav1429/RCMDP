# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 17:57:18 2024

@author: gangu
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os

directory = "C://Users//gangu//OneDrive//Documents//RCMDP//plot_vals"
files = os.listdir(directory)
dic={'name':list(),'leg':[],'start':[]}
baseline = 4
for file in files:
    f = file.split('_')
    if('RS' in f):
        dic['name'].append(file)
        if(f[-1].split('.')[0]=='DQN'):
            dic['start'].append('DQN')
        else:
            dic['start'].append('A2C')
        if('Fast' in f):
            dic['leg'].append("FAST_RSLA")
        elif('no' in f):
            dic['leg'].append('Unconstrained')
        else:
            dic['leg'].append("RSLA")

ax1_leg = []
ax3_leg = []
fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(10,4))
for i,file in enumerate(dic['name']):
    f_path = os.path.join(directory,file)
    data = pd.read_excel(f_path)
    vf,cf = data['vf'],data['cf']
    ax1.plot(cf,label=dic['leg'][i]+"_"+dic['start'][i])
    ax1.set_xlabel('Update Iteration')
    ax1.set_ylabel('Cost to go')
    ax3.plot(vf,label=dic['leg'][i]+"_"+dic['start'][i])
    #ax2.set_title('Reward update for RSLA with robustness guarantee')
    ax3.set_xlabel('Update Iteration')
    ax3.set_ylabel('Value function update')
ax1.plot(np.ones(len(cf))*baseline,linestyle='--',label='Baseline')
ax1.legend(loc='best')
ax3.legend(loc='best')
plt.savefig('RS_plot.pdf')

plt.show()
    

