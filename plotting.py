# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 12:14:07 2024

@author: gangu
"""

#generating plots

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

path = "VF_CF_values_Fast_RSLA_MR_DQN.xlsx"

data = pd.read_excel(path)
vf = data['vf']
cf = data['cf']

baseline  = 3

fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(10,4))
plt.subplots_adjust( right=1)
fig.suptitle('Behaviour of cost and reward for Fast RSLA with robustness guarantee on Machine Replacement')
ax1.plot(cf)
ax1.plot(np.ones(len(cf))*baseline,linestyle='--')
#ax1.set_title('Behaviour of cost for RSLA with robustness guarantee')
ax1.set_xlabel('Update Iteration')
ax1.set_ylabel('Cost to go')
ax3.plot(vf)
#ax2.set_title('Reward update for RSLA with robustness guarantee')
ax3.set_xlabel('Update Iteration')
ax3.set_ylabel('Value function update')
plt.savefig('MR_FAST_RSLA_DQN_started.pdf')

plt.show()

