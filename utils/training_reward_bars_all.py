import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib import rc
import matplotlib

matplotlib.rcParams['font.family'] = 'Nimbus Sans'

# Create figure and subplots
fig, axes = plt.subplots(2, 3, figsize=(13, 7))

import matplotlib.colors as mcolors
import numpy as np

colors = ['#156082', '#7F7F7F', '#00CAD2', '#007EC5', '#00A2E9', '#003F5C','#00E5FF', '#00B89F', '#008CFF','#156082', '#00CAD2', '#007EC5', '#00A2E9', '#003F5C','#00E5FF', '#00B89F', '#008CFF','#156082', '#00CAD2', '#007EC5', '#00A2E9', '#003F5C','#00E5FF', '#00B89F', '#008CFF','#156082', '#00CAD2', '#007EC5', '#00A2E9', '#003F5C','#00E5FF', '#00B89F', '#008CFF']

## Clipping factor

allowed_integers_EoM = np.array([66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 85, 86, 87, 88, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130])

keys = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]) 

rewards_EoM = pd.read_csv('030325_analysis/total_reward.csv')  # Read as numpy array
rewards_EoM = np.array(rewards_EoM.iloc[:,1])

reward_percentile_EoM = pd.read_csv('030325_analysis/total_reward_percentile.csv') 
reward_percentile_EoM = np.array(reward_percentile_EoM.iloc[:,1:])

allowed_integers_CM_CfD = np.array([46, 47, 48, 52, 53, 56, 57, 60, 61, 64, 79, 80, 81, 82, 83, 89, 91, 93, 95, 97, 99, 101, 103, 105, 107, 109, 111]) 
allowed_integers_EoM = allowed_integers_CM_CfD

rewards_CM_CfD = pd.read_csv('040125_analysis/total_reward.csv')  # Read as numpy array
rewards_CM_CfD = np.array(rewards_CM_CfD.iloc[:,1])

reward_percentile_CM_CfD = pd.read_csv('040125_analysis/total_reward_percentile.csv') 
reward_percentile_CM_CfD = np.array(reward_percentile_CM_CfD.iloc[:,1:])

## Clipping factor

# 82, 47, 56, 57

# M.1, M.2, M.3, M.4

row = 0
col = 0

ax = axes[row, col]

run_names = ['M.1', 'M.2', 'M.3', 'M.4']
x = np.arange(len(run_names))
bar_width = 0.4

ax2 = ax.twinx()

ax.set_ylim(-1.25, 12.5)
ax2.set_ylim(-0.25, 2.5)

ax.xaxis.set_major_locator(plt.MaxNLocator(4))
ax.yaxis.set_major_locator(plt.MaxNLocator(4))  
ax2.yaxis.set_major_locator(plt.MaxNLocator(4)) 

# Plot bars with user-specified colors
l1 = ax.bar(x[0] - bar_width/2, rewards_EoM[keys[int(np.where(allowed_integers_EoM == 82)[0])]], bar_width, label=f'EoM', color='white', hatch='xx', edgecolor=colors[0])
l2 = ax2.bar(x[0] + bar_width/2, rewards_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == 82)[0])]], bar_width, label=f'CM + CfD', color=colors[0], edgecolor=colors[0])

ax.errorbar(x[0] - bar_width/2, l1[0].get_height(), yerr = reward_percentile_EoM[keys[int(np.where(allowed_integers_EoM == 82)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)
ax2.errorbar(x[0] + bar_width/2, l2[0].get_height(), yerr = reward_percentile_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == 82)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

# Plot bars with user-specified colors
m1 = ax.bar(x[1] - bar_width/2, rewards_EoM[keys[int(np.where(allowed_integers_EoM == 47)[0])]], bar_width, color='white', hatch='xx', edgecolor=colors[1])
m2 = ax2.bar(x[1] + bar_width/2, rewards_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == 47)[0])]], bar_width, color=colors[1], edgecolor=colors[1])

ax.errorbar(x[1] - bar_width/2, m1[0].get_height(), yerr = reward_percentile_EoM[keys[int(np.where(allowed_integers_EoM == 47)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)
ax2.errorbar(x[1] + bar_width/2, m2[0].get_height(), yerr = reward_percentile_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == 47)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

# Plot bars with user-specified colors
m1 = ax.bar(x[2] - bar_width/2, rewards_EoM[keys[int(np.where(allowed_integers_EoM == 56)[0])]], bar_width, color='white', hatch='xx', edgecolor=colors[2])
m2 = ax2.bar(x[2] + bar_width/2, rewards_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == 56)[0])]], bar_width, color=colors[2], edgecolor=colors[2])

ax.errorbar(x[2] - bar_width/2, m1[0].get_height(), yerr = reward_percentile_EoM[keys[int(np.where(allowed_integers_EoM == 56)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)
ax2.errorbar(x[2] + bar_width/2, m2[0].get_height(), yerr = reward_percentile_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == 56)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

# Plot bars with user-specified colors
m1 = ax.bar(x[3] - bar_width/2, rewards_EoM[keys[int(np.where(allowed_integers_EoM == 57)[0])]], bar_width, color='white', hatch='xx', edgecolor=colors[3])
m2 = ax2.bar(x[3] + bar_width/2, rewards_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == 57)[0])]], bar_width, color=colors[3], edgecolor=colors[3])

ax.errorbar(x[3] - bar_width/2, m1[0].get_height(), yerr = reward_percentile_EoM[keys[int(np.where(allowed_integers_EoM == 57)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)
ax2.errorbar(x[3] + bar_width/2, m2[0].get_height(), yerr = reward_percentile_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == 57)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

ax.axhline(0, color='gray', linestyle='--')

# Labels and title
ax.set_ylabel('Reward EoM [$]', loc='top')
#ax2.set_ylabel('Reward CM + CfD [$]', loc = 'bottom')
ax.set_title('Clipping Factor')
ax.set_xticks(x)
ax.set_xticklabels(run_names)
lns = [l1,l2]
leg = ax2.legend(handles=lns, loc='best')
leg.set_zorder(3)
ax2.set_yticks([])

## Batch size

# 53, 52, 47, 64, 80, 81,

# M.5, M.6, M.2, M.7, M.8, M.9

row = 0
col = 1

ax = axes[row, col]

run_names = ['M.5', 'M.6', 'M.2', 'M.7', 'M.8', 'M.9']
x = np.arange(len(run_names))
bar_width = 0.4

ax2 = ax.twinx()

ax.set_ylim(-1.25, 12.5)
ax2.set_ylim(-0.25, 2.5)

ax.xaxis.set_major_locator(plt.MaxNLocator(4))
ax.yaxis.set_major_locator(plt.MaxNLocator(4))  
ax2.yaxis.set_major_locator(plt.MaxNLocator(4)) 

# Plot bars with user-specified colors
l1 = ax.bar(x[0] - bar_width/2, rewards_EoM[keys[int(np.where(allowed_integers_EoM == 53)[0])]], bar_width, label=f'EoM', color='white', hatch='xx', edgecolor=colors[4])
l2 = ax2.bar(x[0] + bar_width/2, rewards_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == 53)[0])]], bar_width, label=f'CM + CfD', color=colors[4], edgecolor=colors[4])

ax.errorbar(x[0] - bar_width/2, l1[0].get_height(), yerr = reward_percentile_EoM[keys[int(np.where(allowed_integers_EoM == 53)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)
ax2.errorbar(x[0] + bar_width/2, l2[0].get_height(), yerr = reward_percentile_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == 53)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

# Plot bars with user-specified colors
m1 = ax.bar(x[1] - bar_width/2, rewards_EoM[keys[int(np.where(allowed_integers_EoM == 52)[0])]], bar_width, color='white', hatch='xx', edgecolor=colors[5])
m2 = ax2.bar(x[1] + bar_width/2, rewards_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == 52)[0])]], bar_width, color=colors[5], edgecolor=colors[5])

ax.errorbar(x[1] - bar_width/2, m1[0].get_height(), yerr = reward_percentile_EoM[keys[int(np.where(allowed_integers_EoM == 52)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)
ax2.errorbar(x[1] + bar_width/2, m2[0].get_height(), yerr = reward_percentile_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == 52)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

# Plot bars with user-specified colors
m1 = ax.bar(x[2] - bar_width/2, rewards_EoM[keys[int(np.where(allowed_integers_EoM == 47)[0])]], bar_width, color='white', hatch='xx', edgecolor=colors[1])
m2 = ax2.bar(x[2] + bar_width/2, rewards_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == 47)[0])]], bar_width, color=colors[1], edgecolor=colors[1])

ax.errorbar(x[2] - bar_width/2, m1[0].get_height(), yerr = reward_percentile_EoM[keys[int(np.where(allowed_integers_EoM == 47)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)
ax2.errorbar(x[2] + bar_width/2, m2[0].get_height(), yerr = reward_percentile_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == 47)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

# Plot bars with user-specified colors
m1 = ax.bar(x[3] - bar_width/2, rewards_EoM[keys[int(np.where(allowed_integers_EoM == 64)[0])]], bar_width, color='white', hatch='xx', edgecolor=colors[6])
m2 = ax2.bar(x[3] + bar_width/2, rewards_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == 64)[0])]], bar_width, color=colors[6], edgecolor=colors[6])

ax.errorbar(x[3] - bar_width/2, m1[0].get_height(), yerr = reward_percentile_EoM[keys[int(np.where(allowed_integers_EoM == 64)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)
ax2.errorbar(x[3] + bar_width/2, m2[0].get_height(), yerr = reward_percentile_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == 64)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

# Plot bars with user-specified colors
m1 = ax.bar(x[4] - bar_width/2, rewards_EoM[keys[int(np.where(allowed_integers_EoM == 80)[0])]], bar_width, color='white', hatch='xx', edgecolor=colors[7])
m2 = ax2.bar(x[4] + bar_width/2, rewards_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == 80)[0])]], bar_width, color=colors[7], edgecolor=colors[7])

ax.errorbar(x[4] - bar_width/2, m1[0].get_height(), yerr = reward_percentile_EoM[keys[int(np.where(allowed_integers_EoM == 80)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)
ax2.errorbar(x[4] + bar_width/2, m2[0].get_height(), yerr = reward_percentile_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == 80)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

# Plot bars with user-specified colors
m1 = ax.bar(x[5] - bar_width/2, rewards_EoM[keys[int(np.where(allowed_integers_EoM == 81)[0])]], bar_width, color='white', hatch='xx', edgecolor=colors[8])
m2 = ax2.bar(x[5] + bar_width/2, rewards_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == 81)[0])]], bar_width, color=colors[8], edgecolor=colors[8])

ax.errorbar(x[5] - bar_width/2, m1[0].get_height(), yerr = reward_percentile_EoM[keys[int(np.where(allowed_integers_EoM == 81)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)
ax2.errorbar(x[5] + bar_width/2, m2[0].get_height(), yerr = reward_percentile_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == 81)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

ax.axhline(0, color='gray', linestyle='--')

# Labels and title
#ax.set_ylabel('Reward EoM [$]', loc='top')
#ax2.set_ylabel('Reward CM + CfD [$]', loc = 'bottom')
ax.set_title('Batch Size')
ax.set_xticks(x)
ax.set_xticklabels(run_names)
lns = [l1,l2]
leg.set_zorder(3)
ax.set_yticks([])
ax2.set_yticks([])


## Entropy

# M.10, M.2, M.11
# 46, 47, 48

row = 0
col = 2

ax = axes[row, col]

run_names = ['M.10', 'M.2', 'M.11']
x = np.arange(len(run_names))
bar_width = 0.4

ax2 = ax.twinx()

ax.set_ylim(-1.25, 12.5)
ax2.set_ylim(-0.25, 2.5)

ax.xaxis.set_major_locator(plt.MaxNLocator(4))
ax.yaxis.set_major_locator(plt.MaxNLocator(4))  
ax2.yaxis.set_major_locator(plt.MaxNLocator(4)) 

# Plot bars with user-specified colors
l1 = ax.bar(x[0] - bar_width/2, rewards_EoM[keys[int(np.where(allowed_integers_EoM == 46)[0])]], bar_width, label=f'EoM', color='white', hatch='xx', edgecolor=colors[9])
l2 = ax2.bar(x[0] + bar_width/2, rewards_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == 46)[0])]], bar_width, label=f'CM + CfD', color=colors[9], edgecolor=colors[9])

ax.errorbar(x[0] - bar_width/2, l1[0].get_height(), yerr = reward_percentile_EoM[keys[int(np.where(allowed_integers_EoM == 46)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)
ax2.errorbar(x[0] + bar_width/2, l2[0].get_height(), yerr = reward_percentile_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == 46)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

# Plot bars with user-specified colors
m1 = ax.bar(x[1] - bar_width/2, rewards_EoM[keys[int(np.where(allowed_integers_EoM == 47)[0])]], bar_width, color='white', hatch='xx', edgecolor=colors[1])
m2 = ax2.bar(x[1] + bar_width/2, rewards_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == 47)[0])]], bar_width, color=colors[1], edgecolor=colors[1])

ax.errorbar(x[1] - bar_width/2, m1[0].get_height(), yerr = reward_percentile_EoM[keys[int(np.where(allowed_integers_EoM == 47)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)
ax2.errorbar(x[1] + bar_width/2, m2[0].get_height(), yerr = reward_percentile_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == 47)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

# Plot bars with user-specified colors
m1 = ax.bar(x[2] - bar_width/2, rewards_EoM[keys[int(np.where(allowed_integers_EoM == 48)[0])]], bar_width, color='white', hatch='xx', edgecolor=colors[10])
m2 = ax2.bar(x[2] + bar_width/2, rewards_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == 48)[0])]], bar_width, color=colors[10], edgecolor=colors[10])

ax.errorbar(x[2] - bar_width/2, m1[0].get_height(), yerr = reward_percentile_EoM[keys[int(np.where(allowed_integers_EoM == 48)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)
ax2.errorbar(x[2] + bar_width/2, m2[0].get_height(), yerr = reward_percentile_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == 48)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

ax.axhline(0, color='gray', linestyle='--')

# Labels and title
#ax.set_ylabel('Reward EoM [$]', loc='top')
ax2.set_ylabel('Reward CM + CfD [$]', loc = 'bottom')
ax.set_title('Entropy')
ax.set_xticks(x)
ax.set_xticklabels(run_names)
lns = [l1,l2]
ax.set_yticks([])

leg.set_zorder(3)


## MLP Configuration

# M.12, M.2, M.13, M.14
# 60, 47, 83, 61

row = 1
col = 0

ax = axes[row, col]

run_names = ['M.12', 'M.2', 'M.13', 'M.14']
x = np.arange(len(run_names))
bar_width = 0.4

ax2 = ax.twinx()

ax.set_ylim(-1.25, 12.5)
ax2.set_ylim(-0.25, 2.5)

ax.xaxis.set_major_locator(plt.MaxNLocator(4))
ax.yaxis.set_major_locator(plt.MaxNLocator(4))  
ax2.yaxis.set_major_locator(plt.MaxNLocator(4)) 

# Plot bars with user-specified colors
l1 = ax.bar(x[0] - bar_width/2, rewards_EoM[keys[int(np.where(allowed_integers_EoM == 60)[0])]], bar_width, label=f'EoM', color='white', hatch='xx', edgecolor=colors[11])
l2 = ax2.bar(x[0] + bar_width/2, rewards_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == 60)[0])]], bar_width, label=f'CM + CfD', color=colors[11], edgecolor=colors[11])

ax.errorbar(x[0] - bar_width/2, l1[0].get_height(), yerr = reward_percentile_EoM[keys[int(np.where(allowed_integers_EoM == 60)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)
ax2.errorbar(x[0] + bar_width/2, l2[0].get_height(), yerr = reward_percentile_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == 60)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

# Plot bars with user-specified colors
m1 = ax.bar(x[1] - bar_width/2, rewards_EoM[keys[int(np.where(allowed_integers_EoM == 47)[0])]], bar_width, color='white', hatch='xx', edgecolor=colors[1])
m2 = ax2.bar(x[1] + bar_width/2, rewards_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == 47)[0])]], bar_width, color=colors[1], edgecolor=colors[1])

ax.errorbar(x[1] - bar_width/2, m1[0].get_height(), yerr = reward_percentile_EoM[keys[int(np.where(allowed_integers_EoM == 47)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)
ax2.errorbar(x[1] + bar_width/2, m2[0].get_height(), yerr = reward_percentile_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == 47)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

# Plot bars with user-specified colors
m1 = ax.bar(x[2] - bar_width/2, rewards_EoM[keys[int(np.where(allowed_integers_EoM == 83)[0])]], bar_width, color='white', hatch='xx', edgecolor=colors[12])
m2 = ax2.bar(x[2] + bar_width/2, rewards_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == 83)[0])]], bar_width, color=colors[12], edgecolor=colors[12])

ax.errorbar(x[2] - bar_width/2, m1[0].get_height(), yerr = reward_percentile_EoM[keys[int(np.where(allowed_integers_EoM == 83)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)
ax2.errorbar(x[2] + bar_width/2, m2[0].get_height(), yerr = reward_percentile_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == 83)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

# Plot bars with user-specified colors
m1 = ax.bar(x[3] - bar_width/2, rewards_EoM[keys[int(np.where(allowed_integers_EoM == 61)[0])]], bar_width, color='white', hatch='xx', edgecolor=colors[13])
m2 = ax2.bar(x[3] + bar_width/2, rewards_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == 61)[0])]], bar_width, color=colors[13], edgecolor=colors[13])

ax.errorbar(x[3] - bar_width/2, m1[0].get_height(), yerr = reward_percentile_EoM[keys[int(np.where(allowed_integers_EoM == 61)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)
ax2.errorbar(x[3] + bar_width/2, m2[0].get_height(), yerr = reward_percentile_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == 61)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

ax.axhline(0, color='gray', linestyle='--')

# Labels and title
ax.set_ylabel('Reward EoM [$]', loc='top')
#ax2.set_ylabel('Reward CM + CfD [$]', loc = 'bottom')
ax.set_title('MLP Configuration')
ax.set_xticks(x)
ax.set_xticklabels(run_names)
lns = [l1,l2]
ax2.set_yticks([])
leg.set_zorder(3)


## LSTM using MLP in tail

# L.1, L.2, L.3, L.4, L.5, L.6
# 91, 89, 93, 95, 97, 99

row = 1
col = 1

ax = axes[row, col]

run_names = ['L.1', 'L.2', 'L.3', 'L.4', 'L.5', 'L.6']
x = np.arange(len(run_names))
bar_width = 0.4

ax2 = ax.twinx()

ax.set_ylim(-1.25, 12.5)
ax2.set_ylim(-0.25, 2.5)

ax.xaxis.set_major_locator(plt.MaxNLocator(4))
ax.yaxis.set_major_locator(plt.MaxNLocator(4))  
ax2.yaxis.set_major_locator(plt.MaxNLocator(4)) 

# Plot bars with user-specified colors
l1 = ax.bar(x[0] - bar_width/2, rewards_EoM[keys[int(np.where(allowed_integers_EoM == 91)[0])]], bar_width, label=f'EoM', color='white', hatch='xx', edgecolor=colors[14])
l2 = ax2.bar(x[0] + bar_width/2, rewards_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == 91)[0])]], bar_width, label=f'CM + CfD', color=colors[14], edgecolor=colors[14])

ax.errorbar(x[0] - bar_width/2, l1[0].get_height(), yerr = reward_percentile_EoM[keys[int(np.where(allowed_integers_EoM == 91)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)
ax2.errorbar(x[0] + bar_width/2, l2[0].get_height(), yerr = reward_percentile_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == 91)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

# Plot bars with user-specified colors
m1 = ax.bar(x[1] - bar_width/2, rewards_EoM[keys[int(np.where(allowed_integers_EoM == 89)[0])]], bar_width, color='white', hatch='xx', edgecolor=colors[15])
m2 = ax2.bar(x[1] + bar_width/2, rewards_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == 89)[0])]], bar_width, color=colors[15], edgecolor=colors[15])

ax.errorbar(x[1] - bar_width/2, m1[0].get_height(), yerr = reward_percentile_EoM[keys[int(np.where(allowed_integers_EoM == 89)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)
ax2.errorbar(x[1] + bar_width/2, m2[0].get_height(), yerr = reward_percentile_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == 89)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

# Plot bars with user-specified colors
m1 = ax.bar(x[2] - bar_width/2, rewards_EoM[keys[int(np.where(allowed_integers_EoM == 93)[0])]], bar_width, color='white', hatch='xx', edgecolor=colors[16])
m2 = ax2.bar(x[2] + bar_width/2, rewards_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == 93)[0])]], bar_width, color=colors[16], edgecolor=colors[16])

ax.errorbar(x[2] - bar_width/2, m1[0].get_height(), yerr = reward_percentile_EoM[keys[int(np.where(allowed_integers_EoM == 93)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)
ax2.errorbar(x[2] + bar_width/2, m2[0].get_height(), yerr = reward_percentile_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == 93)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

# Plot bars with user-specified colors
m1 = ax.bar(x[3] - bar_width/2, rewards_EoM[keys[int(np.where(allowed_integers_EoM == 95)[0])]], bar_width, color='white', hatch='xx', edgecolor=colors[17])
m2 = ax2.bar(x[3] + bar_width/2, rewards_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == 95)[0])]], bar_width, color=colors[17], edgecolor=colors[17])

ax.errorbar(x[3] - bar_width/2, m1[0].get_height(), yerr = reward_percentile_EoM[keys[int(np.where(allowed_integers_EoM == 95)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)
ax2.errorbar(x[3] + bar_width/2, m2[0].get_height(), yerr = reward_percentile_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == 95)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

# Plot bars with user-specified colors
m1 = ax.bar(x[4] - bar_width/2, rewards_EoM[keys[int(np.where(allowed_integers_EoM == 97)[0])]], bar_width, color='white', hatch='xx', edgecolor=colors[18])
m2 = ax2.bar(x[4] + bar_width/2, rewards_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == 97)[0])]], bar_width, color=colors[18], edgecolor=colors[18])

ax.errorbar(x[4] - bar_width/2, m1[0].get_height(), yerr = reward_percentile_EoM[keys[int(np.where(allowed_integers_EoM == 97)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)
ax2.errorbar(x[4] + bar_width/2, m2[0].get_height(), yerr = reward_percentile_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == 97)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

# Plot bars with user-specified colors
m1 = ax.bar(x[5] - bar_width/2, rewards_EoM[keys[int(np.where(allowed_integers_EoM == 99)[0])]], bar_width, color='white', hatch='xx', edgecolor=colors[19])
m2 = ax2.bar(x[5] + bar_width/2, rewards_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == 99)[0])]], bar_width, color=colors[19], edgecolor=colors[19])

ax.errorbar(x[5] - bar_width/2, m1[0].get_height(), yerr = reward_percentile_EoM[keys[int(np.where(allowed_integers_EoM == 99)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)
ax2.errorbar(x[5] + bar_width/2, m2[0].get_height(), yerr = reward_percentile_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == 99)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

ax.axhline(0, color='gray', linestyle='--')

# Labels and title
#ax.set_ylabel('Reward EoM [$]', loc='top')
#ax2.set_ylabel('Reward CM + CfD [$]', loc = 'bottom')
ax.set_title('LSTM with MLP in tail')
ax.set_xticks(x)
ax.set_xticklabels(run_names)
lns = [l1,l2]
ax.set_yticks([])
ax2.set_yticks([])

leg.set_zorder(3)


## LSTM using MLP in head

# L.7, L.8, L.9, L.10, L.11, L.12
# 103, 101, 105, 107, 109, 111

row = 1
col = 2

ax = axes[row, col]

run_names = ['L.7', 'L.8', 'L.9', 'L.10', 'L.11', 'L.12']
x = np.arange(len(run_names))
bar_width = 0.4

ax2 = ax.twinx()

ax.set_ylim(-1.25, 12.5)
ax2.set_ylim(-0.25, 2.5)

ax.xaxis.set_major_locator(plt.MaxNLocator(4))
ax.yaxis.set_major_locator(plt.MaxNLocator(4))  
ax2.yaxis.set_major_locator(plt.MaxNLocator(4)) 

# Plot bars with user-specified colors
l1 = ax.bar(x[0] - bar_width/2, rewards_EoM[keys[int(np.where(allowed_integers_EoM == 103)[0])]], bar_width, label=f'EoM', color='white', hatch='xx', edgecolor=colors[20])
l2 = ax2.bar(x[0] + bar_width/2, rewards_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == 103)[0])]], bar_width, label=f'CM + CfD', color=colors[20], edgecolor=colors[20], zorder=1)

ax.errorbar(x[0] - bar_width/2, l1[0].get_height(), yerr = reward_percentile_EoM[keys[int(np.where(allowed_integers_EoM == 103)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)
ax2.errorbar(x[0] + bar_width/2, l2[0].get_height(), yerr = reward_percentile_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == 103)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

# Plot bars with user-specified colors
m1 = ax.bar(x[1] - bar_width/2, rewards_EoM[keys[int(np.where(allowed_integers_EoM == 101)[0])]], bar_width,color='white', hatch='xx', edgecolor=colors[21])
m2 = ax2.bar(x[1] + bar_width/2, rewards_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == 101)[0])]], bar_width, color=colors[21], edgecolor=colors[21], zorder=1)

ax.errorbar(x[1] - bar_width/2, m1[0].get_height(), yerr = reward_percentile_EoM[keys[int(np.where(allowed_integers_EoM == 101)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)
ax2.errorbar(x[1] + bar_width/2, m2[0].get_height(), yerr = reward_percentile_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == 101)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

# Plot bars with user-specified colors
m1 = ax.bar(x[2] - bar_width/2, rewards_EoM[keys[int(np.where(allowed_integers_EoM == 105)[0])]], bar_width, color='white', hatch='xx', edgecolor=colors[22])
m2 = ax2.bar(x[2] + bar_width/2, rewards_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == 105)[0])]], bar_width, color=colors[22], edgecolor=colors[22], zorder=1)

ax.errorbar(x[2] - bar_width/2, m1[0].get_height(), yerr = reward_percentile_EoM[keys[int(np.where(allowed_integers_EoM == 105)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)
ax2.errorbar(x[2] + bar_width/2, m2[0].get_height(), yerr = reward_percentile_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == 105)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

# Plot bars with user-specified colors
m1 = ax.bar(x[3] - bar_width/2, rewards_EoM[keys[int(np.where(allowed_integers_EoM == 107)[0])]], bar_width, color='white', hatch='xx', edgecolor=colors[23])
m2 = ax2.bar(x[3] + bar_width/2, rewards_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == 107)[0])]], bar_width, color=colors[23], edgecolor=colors[23], zorder=1)

ax.errorbar(x[3] - bar_width/2, m1[0].get_height(), yerr = reward_percentile_EoM[keys[int(np.where(allowed_integers_EoM == 107)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)
ax2.errorbar(x[3] + bar_width/2, m2[0].get_height(), yerr = reward_percentile_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == 107)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

# Plot bars with user-specified colors
m1 = ax.bar(x[4] - bar_width/2, rewards_EoM[keys[int(np.where(allowed_integers_EoM == 109)[0])]], bar_width, color='white', hatch='xx', edgecolor=colors[24])
m2 = ax2.bar(x[4] + bar_width/2, rewards_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == 109)[0])]], bar_width, color=colors[24], edgecolor=colors[24], zorder=1)

ax.errorbar(x[4] - bar_width/2, m1[0].get_height(), yerr = reward_percentile_EoM[keys[int(np.where(allowed_integers_EoM == 109)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)
ax2.errorbar(x[4] + bar_width/2, m2[0].get_height(), yerr = reward_percentile_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == 109)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

# Plot bars with user-specified colors
m1 = ax.bar(x[5] - bar_width/2, rewards_EoM[keys[int(np.where(allowed_integers_EoM == 111)[0])]], bar_width, color='white', hatch='xx', edgecolor=colors[25])
m2 = ax2.bar(x[5] + bar_width/2, rewards_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == 111)[0])]], bar_width, color=colors[25], edgecolor=colors[25], zorder=1)

ax.errorbar(x[5] - bar_width/2, m1[0].get_height(), yerr = reward_percentile_EoM[keys[int(np.where(allowed_integers_EoM == 111)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)
ax2.errorbar(x[5] + bar_width/2, m2[0].get_height(), yerr = reward_percentile_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == 111)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

ax.axhline(0, color='gray', linestyle='--')

# Labels and title
#ax.set_ylabel('Reward EoM [$]', loc='top')
ax2.set_ylabel('Reward CM + CfD [$]', loc = 'bottom')
ax.set_title('LSTM with MLP in head')
ax.set_xticks(x)
ax.set_xticklabels(run_names)
#ax.set_yticks([])
ax.set_yticks([])


lns = [l1,l2]

leg.set_zorder(3)

# ax.legend(loc='upper left', zorder=3) 
# ax2.legend(loc='upper right', zorder=3)

plt.tight_layout()
plt.show()
plt.savefig("training_reward_bars_all.pdf", format="pdf", bbox_inches="tight")