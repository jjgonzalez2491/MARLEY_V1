import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib import rc
import matplotlib

matplotlib.rcParams['font.family'] = 'Nimbus Sans'

# Create figure and subplots
fig, axes = plt.subplots(1, 6, figsize=(25, 5))
plt.rcParams.update({'font.size': 14})
plt.subplots_adjust(wspace=0, hspace=0)

import matplotlib.colors as mcolors
import numpy as np

colors = ['#156082', '#7F7F7F', '#00CAD2', '#007EC5', '#00A2E9', '#003F5C','#00E5FF', '#00B89F', '#008CFF','#156082', '#00CAD2', '#007EC5', '#00A2E9', '#003F5C','#00E5FF', '#00B89F', '#008CFF','#156082', '#00CAD2', '#007EC5', '#00A2E9', '#003F5C','#00E5FF', '#00B89F', '#008CFF','#156082', '#00CAD2', '#007EC5', '#00A2E9', '#003F5C','#00E5FF', '#00B89F', '#008CFF']

## Clipping factor

allowed_integers = np.array([66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 85, 86, 87, 88, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130])

keys = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]) 

emissions = pd.read_csv('030325_analysis/total_emissions.csv')  # Read as numpy array
emissions = np.array(emissions.iloc[:,1])

emissions_percentile = pd.read_csv('030325_analysis/total_emissions_percentile.csv') 
emissions_percentile = np.array(emissions_percentile.iloc[:,1:])

penalty = pd.read_csv('030325_analysis/total_penalty.csv')  # Read as numpy array
penalty = np.array(penalty.iloc[:,1])

penalty_percentile = pd.read_csv('030325_analysis/total_penalty_percentile.csv') 
penalty_percentile = np.array(penalty_percentile.iloc[:,1:])

HHI_capacity = pd.read_csv('030325_analysis/HHI_capacity_2040.csv')  # Read as numpy array
HHI_capacity = np.array(HHI_capacity.iloc[:,1])

HHI_capacity_percentile = pd.read_csv('030325_analysis/percentile_HHI_capacity_2040.csv') 
HHI_capacity_percentile = np.array(HHI_capacity_percentile.iloc[:,1:])

## Clipping factor

# 87,67,71,72

# AM.1, AM.2, AM.3, AM.4

row = 0
col = 0

ax = axes[col]

run_names = ['AM.1', 'AM.2', 'AM.3', 'AM.4']
x = np.arange(len(run_names))
bar_width = 0.2

ax2 = ax.twinx()
ax3 = ax.twinx()

ax.set_ylim(0, 100)
ax2.set_ylim(1000, 1700)
ax3.set_ylim(0.7, 1.8)

ax.xaxis.set_major_locator(plt.MaxNLocator(4))
ax.yaxis.set_major_locator(plt.MaxNLocator(4))  
ax2.yaxis.set_major_locator(plt.MaxNLocator(4)) 
ax3.yaxis.set_major_locator(plt.MaxNLocator(4)) 

# Plot bars with user-specified colors
l1 = ax.bar(x[0] - bar_width, penalty[keys[int(np.where(allowed_integers == 87)[0])]], bar_width, color='white', hatch='-', label=f'Penalty', edgecolor=colors[0])
ax.errorbar(x[0] - bar_width, l1[0].get_height(), yerr = penalty_percentile[keys[int(np.where(allowed_integers == 87)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

l2 = ax2.bar(x[0], HHI_capacity[keys[int(np.where(allowed_integers == 87)[0])]], bar_width, label=f'HHI', color=colors[0], edgecolor=colors[0])
ax2.errorbar(x[0], l2[0].get_height(), yerr = HHI_capacity_percentile[keys[int(np.where(allowed_integers == 87)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

l3 = ax3.bar(x[0] + bar_width, emissions[keys[int(np.where(allowed_integers == 87)[0])]], bar_width, label=f'Emissions', color='white', hatch='oo', edgecolor=colors[0])
ax3.errorbar(x[0] + bar_width, l3[0].get_height(), yerr = emissions_percentile[keys[int(np.where(allowed_integers == 87)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

# Plot bars with user-specified colors
m1 = ax.bar(x[1] - bar_width, penalty[keys[int(np.where(allowed_integers == 67)[0])]], bar_width, color='white', hatch='-',  edgecolor=colors[1])
ax.errorbar(x[1] - bar_width, m1[0].get_height(), yerr = penalty_percentile[keys[int(np.where(allowed_integers == 67)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

m2 = ax2.bar(x[1], HHI_capacity[keys[int(np.where(allowed_integers == 67)[0])]], bar_width, color=colors[1], edgecolor=colors[1])
ax2.errorbar(x[1], m2[0].get_height(), yerr = HHI_capacity_percentile[keys[int(np.where(allowed_integers == 67)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

m3 = ax3.bar(x[1] + bar_width, emissions[keys[int(np.where(allowed_integers == 67)[0])]], bar_width, color='white', hatch='oo', edgecolor=colors[1])
ax3.errorbar(x[1] + bar_width, m3[0].get_height(), yerr = emissions_percentile[keys[int(np.where(allowed_integers == 67)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

# Plot bars with user-specified colors
m1 = ax.bar(x[2] - bar_width, penalty[keys[int(np.where(allowed_integers == 71)[0])]], bar_width, color='white', hatch='-', edgecolor=colors[2])
ax.errorbar(x[2] - bar_width, m1[0].get_height(), yerr = penalty_percentile[keys[int(np.where(allowed_integers == 71)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

m2 = ax2.bar(x[2], HHI_capacity[keys[int(np.where(allowed_integers == 71)[0])]], bar_width, color=colors[2], edgecolor=colors[2])
ax2.errorbar(x[2], m2[0].get_height(), yerr = HHI_capacity_percentile[keys[int(np.where(allowed_integers == 71)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

m3 = ax3.bar(x[2] + bar_width, emissions[keys[int(np.where(allowed_integers == 71)[0])]], bar_width, color='white', hatch='oo', edgecolor=colors[2])
ax3.errorbar(x[2] + bar_width, m3[0].get_height(), yerr = emissions_percentile[keys[int(np.where(allowed_integers == 71)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

# Plot bars with user-specified colors
m1 = ax.bar(x[3] - bar_width, penalty[keys[int(np.where(allowed_integers == 72)[0])]], bar_width, color='white', hatch='-', edgecolor=colors[3])
ax.errorbar(x[3] - bar_width, m1[0].get_height(), yerr = penalty_percentile[keys[int(np.where(allowed_integers == 72)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

m2 = ax2.bar(x[3], HHI_capacity[keys[int(np.where(allowed_integers == 72)[0])]], bar_width, color=colors[3], edgecolor=colors[3])
ax2.errorbar(x[3], m2[0].get_height(), yerr = HHI_capacity_percentile[keys[int(np.where(allowed_integers == 72)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

m3 = ax3.bar(x[3] + bar_width, emissions[keys[int(np.where(allowed_integers == 72)[0])]], bar_width, color='white', hatch='oo', edgecolor=colors[3])
ax3.errorbar(x[3] + bar_width, m3[0].get_height(), yerr = emissions_percentile[keys[int(np.where(allowed_integers == 72)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

# Labels and title
ax.set_ylabel('Penalty [$]', loc='top')
#ax2.set_ylabel('HHI [$]', loc = 'bottom')
ax.set_title('Clipping Factor')
ax.set_xticks(x)
ax.set_xticklabels(run_names)
lns = [l1,l2, l3]
leg = ax2.legend(handles=lns, loc='upper left')
leg.set_zorder(3)
ax3.get_yaxis().set_visible(False)
ax2.set_yticks([])


## Batch size

# 70,69,67,75,85,86

# AM.5, AM.6, AM.2, AM.7, AM.8, AM.9

row = 0
col = 1

ax = axes[col]

run_names = ['AM.5', 'AM.6', 'AM.2', 'AM.7', 'AM.8', 'AM.9']
x = np.arange(len(run_names))

bar_width = 0.2

ax2 = ax.twinx()
ax3 = ax.twinx()

ax.set_ylim(0, 100)
ax2.set_ylim(1000, 1700)
ax3.set_ylim(0.7, 1.8)
ax.xaxis.set_major_locator(plt.MaxNLocator(4))
ax.yaxis.set_major_locator(plt.MaxNLocator(4))  
ax2.yaxis.set_major_locator(plt.MaxNLocator(4)) 
ax3.yaxis.set_major_locator(plt.MaxNLocator(4)) 

# Plot bars with user-specified colors
l1 = ax.bar(x[0] - bar_width, penalty[keys[int(np.where(allowed_integers == 70)[0])]], bar_width, label=f'Penalty', color='white', hatch='-',  edgecolor=colors[4])
ax.errorbar(x[0] - bar_width, l1[0].get_height(), yerr = penalty_percentile[keys[int(np.where(allowed_integers == 70)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

l2 = ax2.bar(x[0], HHI_capacity[keys[int(np.where(allowed_integers == 70)[0])]], bar_width, label=f'HHI', color=colors[4], edgecolor=colors[4])
ax2.errorbar(x[0], l2[0].get_height(), yerr = HHI_capacity_percentile[keys[int(np.where(allowed_integers == 70)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

l3 = ax3.bar(x[0] + bar_width, emissions[keys[int(np.where(allowed_integers == 70)[0])]], bar_width, label=f'Emissions', color='white', hatch='oo', edgecolor=colors[4])
ax3.errorbar(x[0] + bar_width, l3[0].get_height(), yerr = emissions_percentile[keys[int(np.where(allowed_integers == 70)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

# Plot bars with user-specified colors
m1 = ax.bar(x[1] - bar_width, penalty[keys[int(np.where(allowed_integers == 69)[0])]], bar_width, color='white', hatch='-',  edgecolor=colors[5])
ax.errorbar(x[1] - bar_width, m1[0].get_height(), yerr = penalty_percentile[keys[int(np.where(allowed_integers == 69)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

m2 = ax2.bar(x[1], HHI_capacity[keys[int(np.where(allowed_integers == 69)[0])]], bar_width, color=colors[5], edgecolor=colors[5])
ax2.errorbar(x[1], m2[0].get_height(), yerr = HHI_capacity_percentile[keys[int(np.where(allowed_integers == 69)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

m3 = ax3.bar(x[1] + bar_width, emissions[keys[int(np.where(allowed_integers == 69)[0])]], bar_width, color='white', hatch='oo', edgecolor=colors[5])
ax3.errorbar(x[1] + bar_width, m3[0].get_height(), yerr = emissions_percentile[keys[int(np.where(allowed_integers == 69)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

# Plot bars with user-specified colors
m1 = ax.bar(x[2] - bar_width, penalty[keys[int(np.where(allowed_integers == 67)[0])]], bar_width, color='white', hatch='-', edgecolor=colors[1])
ax.errorbar(x[2] - bar_width, m1[0].get_height(), yerr = penalty_percentile[keys[int(np.where(allowed_integers == 67)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

m2 = ax2.bar(x[2], HHI_capacity[keys[int(np.where(allowed_integers == 67)[0])]], bar_width, color=colors[1], edgecolor=colors[1])
ax2.errorbar(x[2], m2[0].get_height(), yerr = HHI_capacity_percentile[keys[int(np.where(allowed_integers == 67)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

m3 = ax3.bar(x[2] + bar_width, emissions[keys[int(np.where(allowed_integers == 67)[0])]], bar_width, color='white', hatch='oo', edgecolor=colors[1])
ax3.errorbar(x[2] + bar_width, m3[0].get_height(), yerr = emissions_percentile[keys[int(np.where(allowed_integers == 67)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

# Plot bars with user-specified colors
m1 = ax.bar(x[3] - bar_width, penalty[keys[int(np.where(allowed_integers == 75)[0])]], bar_width, color='white', hatch='-', edgecolor=colors[6])
ax.errorbar(x[3] - bar_width, m1[0].get_height(), yerr = penalty_percentile[keys[int(np.where(allowed_integers ==  75)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

m2 = ax2.bar(x[3], HHI_capacity[keys[int(np.where(allowed_integers ==  75)[0])]], bar_width, color=colors[6], edgecolor=colors[6])
ax2.errorbar(x[3], m2[0].get_height(), yerr = HHI_capacity_percentile[keys[int(np.where(allowed_integers ==  75)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

m3 = ax3.bar(x[3] + bar_width, emissions[keys[int(np.where(allowed_integers ==  75)[0])]], bar_width, color='white', hatch='oo', edgecolor=colors[6])
ax3.errorbar(x[3] + bar_width, m3[0].get_height(), yerr = emissions_percentile[keys[int(np.where(allowed_integers ==  75)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

# Plot bars with user-specified colors
m1 = ax.bar(x[4] - bar_width, penalty[keys[int(np.where(allowed_integers ==  85)[0])]], bar_width, color='white', hatch='-', edgecolor=colors[7])
ax.errorbar(x[4] - bar_width, m1[0].get_height(), yerr = penalty_percentile[keys[int(np.where(allowed_integers ==  85)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

m2 = ax2.bar(x[4], HHI_capacity[keys[int(np.where(allowed_integers ==  85)[0])]], bar_width, color=colors[7], edgecolor=colors[7])
ax2.errorbar(x[4], m2[0].get_height(), yerr = HHI_capacity_percentile[keys[int(np.where(allowed_integers ==  85)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

m3 = ax3.bar(x[4] + bar_width, emissions[keys[int(np.where(allowed_integers ==  85)[0])]], bar_width, color='white', hatch='oo', edgecolor=colors[7])
ax3.errorbar(x[4] + bar_width, m3[0].get_height(), yerr = emissions_percentile[keys[int(np.where(allowed_integers ==  85)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

# Plot bars with user-specified colors
m1 = ax.bar(x[5] - bar_width, penalty[keys[int(np.where(allowed_integers ==  86)[0])]], bar_width, color='white', hatch='-',  edgecolor=colors[8])
ax.errorbar(x[5] - bar_width, m1[0].get_height(), yerr = penalty_percentile[keys[int(np.where(allowed_integers ==  86)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

m2 = ax2.bar(x[5], HHI_capacity[keys[int(np.where(allowed_integers ==  86)[0])]], bar_width, color=colors[8], edgecolor=colors[8])
ax2.errorbar(x[5], m2[0].get_height(), yerr = HHI_capacity_percentile[keys[int(np.where(allowed_integers ==  86)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

m3 = ax3.bar(x[5] + bar_width, emissions[keys[int(np.where(allowed_integers ==  86)[0])]], bar_width, color='white', hatch='oo', edgecolor=colors[8])
ax3.errorbar(x[5] + bar_width, m3[0].get_height(), yerr = emissions_percentile[keys[int(np.where(allowed_integers ==  86)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

# Labels and title
#ax.set_ylabel('Penalty [$]', loc='top')
#ax2.set_ylabel('HHI [$]', loc = 'bottom')
ax.set_title('Batch Size')
ax.set_xticks(x)
ax.set_xticklabels(run_names)
ax3.get_yaxis().set_visible(False)
ax.set_yticks([])
ax2.set_yticks([])



## Entropy

# AM.10, AM.2, AM.11
# 66,67,68

row = 0
col = 2

ax = axes[col]

run_names = ['AM.10', 'AM.2', 'AM.11']
x = np.arange(len(run_names))
bar_width = 0.2

ax2 = ax.twinx()
ax3 = ax.twinx()

ax.set_ylim(0, 100)
ax2.set_ylim(1000, 1700)
ax3.set_ylim(0.7, 1.8)

ax.xaxis.set_major_locator(plt.MaxNLocator(4))
ax.yaxis.set_major_locator(plt.MaxNLocator(4))  
ax2.yaxis.set_major_locator(plt.MaxNLocator(4)) 
ax3.yaxis.set_major_locator(plt.MaxNLocator(4)) 

# Plot bars with user-specified colors
l1 = ax.bar(x[0] - bar_width, penalty[keys[int(np.where(allowed_integers ==  66)[0])]], bar_width, label=f'Penalty',color='white', hatch='-',  edgecolor=colors[9])
ax.errorbar(x[0] - bar_width, l1[0].get_height(), yerr = penalty_percentile[keys[int(np.where(allowed_integers ==  66)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

l2 = ax2.bar(x[0], HHI_capacity[keys[int(np.where(allowed_integers ==  66)[0])]], bar_width, label=f'HHI', color=colors[9], edgecolor=colors[9])
ax2.errorbar(x[0], l2[0].get_height(), yerr = HHI_capacity_percentile[keys[int(np.where(allowed_integers ==  66)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

l3 = ax3.bar(x[0] + bar_width, emissions[keys[int(np.where(allowed_integers ==  66)[0])]], bar_width, label=f'Emissions', color='white', hatch='oo', edgecolor=colors[9])
ax3.errorbar(x[0] + bar_width, l3[0].get_height(), yerr = emissions_percentile[keys[int(np.where(allowed_integers ==  66)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

# Plot bars with user-specified colors
m1 = ax.bar(x[1] - bar_width, penalty[keys[int(np.where(allowed_integers == 67)[0])]], bar_width, color='white', hatch='-', edgecolor=colors[1])
ax.errorbar(x[1] - bar_width, m1[0].get_height(), yerr = penalty_percentile[keys[int(np.where(allowed_integers == 67)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

m2 = ax2.bar(x[1], HHI_capacity[keys[int(np.where(allowed_integers == 67)[0])]], bar_width, color=colors[1], edgecolor=colors[1])
ax2.errorbar(x[1], m2[0].get_height(), yerr = HHI_capacity_percentile[keys[int(np.where(allowed_integers == 67)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

m3 = ax3.bar(x[1] + bar_width, emissions[keys[int(np.where(allowed_integers == 67)[0])]], bar_width, color='white', hatch='oo', edgecolor=colors[1])
ax3.errorbar(x[1] + bar_width, m3[0].get_height(), yerr = emissions_percentile[keys[int(np.where(allowed_integers == 67)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

# Plot bars with user-specified colors
m1 = ax.bar(x[2] - bar_width, penalty[keys[int(np.where(allowed_integers ==  68)[0])]], bar_width, color='white', hatch='-',  edgecolor=colors[10])
ax.errorbar(x[2] - bar_width, m1[0].get_height(), yerr = penalty_percentile[keys[int(np.where(allowed_integers ==  68)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

m2 = ax2.bar(x[2], HHI_capacity[keys[int(np.where(allowed_integers ==  68)[0])]], bar_width, color=colors[10], edgecolor=colors[10])
ax2.errorbar(x[2], m2[0].get_height(), yerr = HHI_capacity_percentile[keys[int(np.where(allowed_integers ==  68)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

m3 = ax3.bar(x[2] + bar_width, emissions[keys[int(np.where(allowed_integers ==  68)[0])]], bar_width, color='white', hatch='oo', edgecolor=colors[10])
ax3.errorbar(x[2] + bar_width, m3[0].get_height(), yerr = emissions_percentile[keys[int(np.where(allowed_integers ==  68)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

# Labels and title
#ax.set_ylabel('Penalty [$]', loc='top')
ax.set_title('Entropy')
ax.set_xticks(x)
ax.set_xticklabels(run_names)
ax3.get_yaxis().set_visible(False)
ax.set_yticks([])
ax2.set_yticks([])


## MLP Configuration

# AM.12, AM.2, AM.13, AM.14
# 73,67,88,74

row = 0
col = 3

ax = axes[col]

run_names = ['AM.12', 'AM.2', 'AM.13', 'AM.14']
x = np.arange(len(run_names))
bar_width = 0.2

ax2 = ax.twinx()
ax3 = ax.twinx()

ax.set_ylim(0, 100)
ax2.set_ylim(1000, 1700)
ax3.set_ylim(0.7, 1.8)

ax.xaxis.set_major_locator(plt.MaxNLocator(4))
ax.yaxis.set_major_locator(plt.MaxNLocator(4))  
ax2.yaxis.set_major_locator(plt.MaxNLocator(4)) 
ax3.yaxis.set_major_locator(plt.MaxNLocator(4)) 

# Plot bars with user-specified colors
l1 = ax.bar(x[0] - bar_width, penalty[keys[int(np.where(allowed_integers == 73)[0])]], bar_width, label=f'Penalty', color='white', hatch='-',  edgecolor=colors[11])
ax.errorbar(x[0] - bar_width, l1[0].get_height(), yerr = penalty_percentile[keys[int(np.where(allowed_integers == 73)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

l2 = ax2.bar(x[0], HHI_capacity[keys[int(np.where(allowed_integers == 73)[0])]], bar_width, label=f'HHI', color=colors[11], edgecolor=colors[11])
ax2.errorbar(x[0], l2[0].get_height(), yerr = HHI_capacity_percentile[keys[int(np.where(allowed_integers == 73)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

l3 = ax3.bar(x[0] + bar_width, emissions[keys[int(np.where(allowed_integers == 73)[0])]], bar_width, label=f'Emissions', color='white', hatch='oo', edgecolor=colors[11])
ax3.errorbar(x[0] + bar_width, l3[0].get_height(), yerr = emissions_percentile[keys[int(np.where(allowed_integers == 73)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

# Plot bars with user-specified colors
m1 = ax.bar(x[1] - bar_width, penalty[keys[int(np.where(allowed_integers == 67)[0])]], bar_width, color='white', hatch='-',  edgecolor=colors[1])
ax.errorbar(x[1] - bar_width, m1[0].get_height(), yerr = penalty_percentile[keys[int(np.where(allowed_integers == 67)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

m2 = ax2.bar(x[1], HHI_capacity[keys[int(np.where(allowed_integers == 67)[0])]], bar_width, color=colors[1], edgecolor=colors[1])
ax2.errorbar(x[1], m2[0].get_height(), yerr = HHI_capacity_percentile[keys[int(np.where(allowed_integers == 67)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

m3 = ax3.bar(x[1] + bar_width, emissions[keys[int(np.where(allowed_integers == 67)[0])]], bar_width, color='white', hatch='oo', edgecolor=colors[1])
ax3.errorbar(x[1] + bar_width, m3[0].get_height(), yerr = emissions_percentile[keys[int(np.where(allowed_integers == 67)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

# Plot bars with user-specified colors
m1 = ax.bar(x[2] - bar_width, penalty[keys[int(np.where(allowed_integers ==  88)[0])]], bar_width, color='white', hatch='-',  edgecolor=colors[12])
ax.errorbar(x[2] - bar_width, m1[0].get_height(), yerr = penalty_percentile[keys[int(np.where(allowed_integers ==  88)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

m2 = ax2.bar(x[2], HHI_capacity[keys[int(np.where(allowed_integers ==  88)[0])]], bar_width, color=colors[12], edgecolor=colors[12])
ax2.errorbar(x[2], m2[0].get_height(), yerr = HHI_capacity_percentile[keys[int(np.where(allowed_integers ==  88)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

m3 = ax3.bar(x[2] + bar_width, emissions[keys[int(np.where(allowed_integers ==  88)[0])]], bar_width, color='white', hatch='oo', edgecolor=colors[12])
ax3.errorbar(x[2] + bar_width, m3[0].get_height(), yerr = emissions_percentile[keys[int(np.where(allowed_integers ==  88)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

# Plot bars with user-specified colors
m1 = ax.bar(x[3] - bar_width, penalty[keys[int(np.where(allowed_integers ==  74)[0])]], bar_width, color='white', hatch='-',  edgecolor=colors[13])
ax.errorbar(x[3] - bar_width, m1[0].get_height(), yerr = penalty_percentile[keys[int(np.where(allowed_integers ==  74)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

m2 = ax2.bar(x[3], HHI_capacity[keys[int(np.where(allowed_integers ==  74)[0])]], bar_width, color=colors[13], edgecolor=colors[13])
ax2.errorbar(x[3], m2[0].get_height(), yerr = HHI_capacity_percentile[keys[int(np.where(allowed_integers ==  74)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

m3 = ax3.bar(x[3] + bar_width, emissions[keys[int(np.where(allowed_integers ==  74)[0])]], bar_width, color='white', hatch='oo', edgecolor=colors[13])
ax3.errorbar(x[3] + bar_width, m3[0].get_height(), yerr = emissions_percentile[keys[int(np.where(allowed_integers ==  74)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

#ax2.set_ylabel('HHI [$]', loc = 'bottom')
ax.set_title('MLP Configuration')
ax.set_xticks(x)
ax.set_xticklabels(run_names)
ax3.get_yaxis().set_visible(False)
ax.set_yticks([])
ax2.set_yticks([])


## LSTM using MLP in tail

# AL.1, AL.2, AL.3, AL.4, AL.5, AL.6
# 120, 119, 121, 122, 123, 124

row = 0
col = 4

ax = axes[col]

run_names = ['AL.1', 'AL.2', 'AL.3', 'AL.4', 'AL.5', 'AL.6']
x = np.arange(len(run_names))
bar_width = 0.2

ax2 = ax.twinx()
ax3 = ax.twinx()

ax.set_ylim(0, 100)
ax2.set_ylim(1000, 1700)
ax3.set_ylim(0.7, 1.8)

ax.xaxis.set_major_locator(plt.MaxNLocator(4))
ax.yaxis.set_major_locator(plt.MaxNLocator(4))  
ax2.yaxis.set_major_locator(plt.MaxNLocator(4)) 
ax3.yaxis.set_major_locator(plt.MaxNLocator(4)) 

# Plot bars with user-specified colors
l1 = ax.bar(x[0] - bar_width, penalty[keys[int(np.where(allowed_integers ==  120)[0])]], bar_width, label=f'Penalty', color='white', hatch='-',  edgecolor=colors[14])
ax.errorbar(x[0] - bar_width, l1[0].get_height(), yerr = penalty_percentile[keys[int(np.where(allowed_integers ==  120)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

l2 = ax2.bar(x[0], HHI_capacity[keys[int(np.where(allowed_integers ==  120)[0])]], bar_width, label=f'HHI', color=colors[14], edgecolor=colors[14])
ax2.errorbar(x[0], l2[0].get_height(), yerr = HHI_capacity_percentile[keys[int(np.where(allowed_integers ==  120)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

l3 = ax3.bar(x[0] + bar_width, emissions[keys[int(np.where(allowed_integers ==  120)[0])]], bar_width, label=f'Emissions', color='white', hatch='oo', edgecolor=colors[14])
ax3.errorbar(x[0] + bar_width, l3[0].get_height(), yerr = emissions_percentile[keys[int(np.where(allowed_integers ==  120)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

# Plot bars with user-specified colors
m1 = ax.bar(x[1] - bar_width, penalty[keys[int(np.where(allowed_integers ==  119)[0])]], bar_width, color='white', hatch='-',  edgecolor=colors[15])
ax.errorbar(x[1] - bar_width, m1[0].get_height(), yerr = penalty_percentile[keys[int(np.where(allowed_integers ==  119)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

m2 = ax2.bar(x[1], HHI_capacity[keys[int(np.where(allowed_integers ==  119)[0])]], bar_width, color=colors[15], edgecolor=colors[15])
ax2.errorbar(x[1], m2[0].get_height(), yerr = HHI_capacity_percentile[keys[int(np.where(allowed_integers ==  119)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

m3 = ax3.bar(x[1] + bar_width, emissions[keys[int(np.where(allowed_integers ==  119)[0])]], bar_width, color='white', hatch='oo', edgecolor=colors[15])
ax3.errorbar(x[1] + bar_width, m3[0].get_height(), yerr = emissions_percentile[keys[int(np.where(allowed_integers ==  119)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

# Plot bars with user-specified colors
m1 = ax.bar(x[2] - bar_width, penalty[keys[int(np.where(allowed_integers ==  121)[0])]], bar_width, color='white', hatch='-',  edgecolor=colors[16])
ax.errorbar(x[2] - bar_width, m1[0].get_height(), yerr = penalty_percentile[keys[int(np.where(allowed_integers ==  121)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

m2 = ax2.bar(x[2], HHI_capacity[keys[int(np.where(allowed_integers ==  121)[0])]], bar_width, color=colors[16], edgecolor=colors[16])
ax2.errorbar(x[2], m2[0].get_height(), yerr = HHI_capacity_percentile[keys[int(np.where(allowed_integers ==  121)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

m3 = ax3.bar(x[2] + bar_width, emissions[keys[int(np.where(allowed_integers ==  121)[0])]], bar_width, color='white', hatch='oo', edgecolor=colors[16])
ax3.errorbar(x[2] + bar_width, m3[0].get_height(), yerr = emissions_percentile[keys[int(np.where(allowed_integers ==  121)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

# Plot bars with user-specified colors
m1 = ax.bar(x[3] - bar_width, penalty[keys[int(np.where(allowed_integers ==  122)[0])]], bar_width, color='white', hatch='-',  edgecolor=colors[17])
ax.errorbar(x[3] - bar_width, m1[0].get_height(), yerr = penalty_percentile[keys[int(np.where(allowed_integers ==  122)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

m2 = ax2.bar(x[3], HHI_capacity[keys[int(np.where(allowed_integers ==  122)[0])]], bar_width, color=colors[17], edgecolor=colors[17])
ax2.errorbar(x[3], m2[0].get_height(), yerr = HHI_capacity_percentile[keys[int(np.where(allowed_integers ==  122)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

m3 = ax3.bar(x[3] + bar_width, emissions[keys[int(np.where(allowed_integers ==  122)[0])]], bar_width, color='white', hatch='oo', edgecolor=colors[17])
ax3.errorbar(x[3] + bar_width, m3[0].get_height(), yerr = emissions_percentile[keys[int(np.where(allowed_integers ==  122)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

# Plot bars with user-specified colors
m1 = ax.bar(x[4] - bar_width, penalty[keys[int(np.where(allowed_integers ==  123)[0])]], bar_width, color='white', hatch='-',  edgecolor=colors[18])
ax.errorbar(x[4] - bar_width, m1[0].get_height(), yerr = penalty_percentile[keys[int(np.where(allowed_integers ==  123)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

m2 = ax2.bar(x[4], HHI_capacity[keys[int(np.where(allowed_integers ==  123)[0])]], bar_width, color=colors[18], edgecolor=colors[18])
ax2.errorbar(x[4], m2[0].get_height(), yerr = HHI_capacity_percentile[keys[int(np.where(allowed_integers ==  123)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

m3 = ax3.bar(x[4] + bar_width, emissions[keys[int(np.where(allowed_integers ==  123)[0])]], bar_width, color='white', hatch='oo', edgecolor=colors[18])
ax3.errorbar(x[4] + bar_width, m3[0].get_height(), yerr = emissions_percentile[keys[int(np.where(allowed_integers ==  123)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

# Plot bars with user-specified colors
m1 = ax.bar(x[5] - bar_width, penalty[keys[int(np.where(allowed_integers ==  124)[0])]], bar_width, color='white', hatch='-',  edgecolor=colors[19])
ax.errorbar(x[5] - bar_width, m1[0].get_height(), yerr = penalty_percentile[keys[int(np.where(allowed_integers ==  124)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

m2 = ax2.bar(x[5], HHI_capacity[keys[int(np.where(allowed_integers ==  124)[0])]], bar_width, color=colors[19], edgecolor=colors[19])
ax2.errorbar(x[5], m2[0].get_height(), yerr = HHI_capacity_percentile[keys[int(np.where(allowed_integers ==  124)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

m3 = ax3.bar(x[5] + bar_width, emissions[keys[int(np.where(allowed_integers ==  124)[0])]], bar_width, color='white', hatch='oo', edgecolor=colors[19])
ax3.errorbar(x[5] + bar_width, m3[0].get_height(), yerr = emissions_percentile[keys[int(np.where(allowed_integers ==  124)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

# Labels and title
#ax.set_ylabel('Penalty [$]', loc='top')
ax3.get_yaxis().set_visible(False)
ax.set_title('LSTM with MLP in tail')
ax.set_xticks(x)
ax.set_xticklabels(run_names)
ax.set_yticks([])
ax2.set_yticks([])



## LSTM using MLP in head

# AL.7, AL.8, AL.9, AL.10, AL.11, AL.12
# 126, 125, 127, 128, 129, 130

row = 0
col = 5

ax = axes[col]

run_names = ['AL.7', 'AL.8', 'AL.9', 'AL.10', 'AL.11', 'AL.12']
x = np.arange(len(run_names))
bar_width = 0.2

ax2 = ax.twinx()
ax3 = ax.twinx()

ax.set_ylim(0, 100)
ax2.set_ylim(1000, 1700)
ax3.set_ylim(0.7, 1.8)

ax.xaxis.set_major_locator(plt.MaxNLocator(4))
ax.yaxis.set_major_locator(plt.MaxNLocator(4))  
ax2.yaxis.set_major_locator(plt.MaxNLocator(4)) 
ax3.yaxis.set_major_locator(plt.MaxNLocator(4)) 

# Plot bars with user-specified colors
l1 = ax.bar(x[0] - bar_width, penalty[keys[int(np.where(allowed_integers ==  126)[0])]], bar_width, label=f'Penalty',color='white', hatch='-',  zorder=1, edgecolor=colors[20])
ax.errorbar(x[0] - bar_width, l1[0].get_height(), yerr = penalty_percentile[keys[int(np.where(allowed_integers ==  126)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

l2 = ax2.bar(x[0], HHI_capacity[keys[int(np.where(allowed_integers ==  126)[0])]], bar_width, label=f'HHI', color=colors[20], edgecolor=colors[20], zorder=1)
ax2.errorbar(x[0], l2[0].get_height(), yerr = HHI_capacity_percentile[keys[int(np.where(allowed_integers ==  126)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

l3 = ax3.bar(x[0] + bar_width, emissions[keys[int(np.where(allowed_integers ==  126)[0])]], bar_width, label=f'Emissions', color='white', hatch='oo', edgecolor=colors[20])
ax3.errorbar(x[0] + bar_width, l3[0].get_height(), yerr = emissions_percentile[keys[int(np.where(allowed_integers ==  126)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

# Plot bars with user-specified colors
m1 = ax.bar(x[1] - bar_width, penalty[keys[int(np.where(allowed_integers ==  125)[0])]], bar_width, color='white', hatch='-',  zorder=1, edgecolor=colors[21])
ax.errorbar(x[1] - bar_width, m1[0].get_height(), yerr = penalty_percentile[keys[int(np.where(allowed_integers ==  125)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

m2 = ax2.bar(x[1], HHI_capacity[keys[int(np.where(allowed_integers ==  125)[0])]], bar_width, color=colors[21], edgecolor=colors[21], zorder=1)
ax2.errorbar(x[1], m2[0].get_height(), yerr = HHI_capacity_percentile[keys[int(np.where(allowed_integers ==  125)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

m3 = ax3.bar(x[1] + bar_width, emissions[keys[int(np.where(allowed_integers ==  125)[0])]], bar_width, color='white', hatch='oo', edgecolor=colors[21])
ax3.errorbar(x[1] + bar_width, m3[0].get_height(), yerr = emissions_percentile[keys[int(np.where(allowed_integers ==  125)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

# Plot bars with user-specified colors
m1 = ax.bar(x[2] - bar_width, penalty[keys[int(np.where(allowed_integers ==  127)[0])]], bar_width, color='white', hatch='-',  zorder=1, edgecolor=colors[22])
ax.errorbar(x[2] - bar_width, m1[0].get_height(), yerr = penalty_percentile[keys[int(np.where(allowed_integers ==  127)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

m2 = ax2.bar(x[2], HHI_capacity[keys[int(np.where(allowed_integers ==  127)[0])]], bar_width, color=colors[22], edgecolor=colors[22], zorder=1)
ax2.errorbar(x[2], m2[0].get_height(), yerr = HHI_capacity_percentile[keys[int(np.where(allowed_integers ==  127)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

m3 = ax3.bar(x[2] + bar_width, emissions[keys[int(np.where(allowed_integers ==  127)[0])]], bar_width, color='white', hatch='oo', edgecolor=colors[22])
ax3.errorbar(x[2] + bar_width, m3[0].get_height(), yerr = emissions_percentile[keys[int(np.where(allowed_integers ==  127)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

# Plot bars with user-specified colors
m1 = ax.bar(x[3] - bar_width, penalty[keys[int(np.where(allowed_integers ==  128)[0])]], bar_width, color='white', hatch='-',  zorder=1, edgecolor=colors[23])
ax.errorbar(x[3] - bar_width, m1[0].get_height(), yerr = penalty_percentile[keys[int(np.where(allowed_integers ==  128)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

m2 = ax2.bar(x[3], HHI_capacity[keys[int(np.where(allowed_integers ==  128)[0])]], bar_width, color=colors[23], edgecolor=colors[23], zorder=1)
ax2.errorbar(x[3], m2[0].get_height(), yerr = HHI_capacity_percentile[keys[int(np.where(allowed_integers ==  128)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

m3 = ax3.bar(x[3] + bar_width, emissions[keys[int(np.where(allowed_integers ==  128)[0])]], bar_width, color='white', hatch='oo', edgecolor=colors[23])
ax3.errorbar(x[3] + bar_width, m3[0].get_height(), yerr = emissions_percentile[keys[int(np.where(allowed_integers ==  128)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

# Plot bars with user-specified colors
m1 = ax.bar(x[4] - bar_width, penalty[keys[int(np.where(allowed_integers ==  129)[0])]], bar_width, color='white', hatch='-', zorder=1, edgecolor=colors[24])
ax.errorbar(x[4] - bar_width, m1[0].get_height(), yerr = penalty_percentile[keys[int(np.where(allowed_integers ==  129)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

m2 =ax2.bar(x[4], HHI_capacity[keys[int(np.where(allowed_integers ==  129)[0])]], bar_width, color=colors[24], edgecolor=colors[24], zorder=1)
ax2.errorbar(x[4], m2[0].get_height(), yerr = HHI_capacity_percentile[keys[int(np.where(allowed_integers ==  129)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

m3 = ax3.bar(x[4] + bar_width, emissions[keys[int(np.where(allowed_integers ==  129)[0])]], bar_width, color='white', hatch='oo', edgecolor=colors[24])
ax3.errorbar(x[4] + bar_width, m3[0].get_height(), yerr = emissions_percentile[keys[int(np.where(allowed_integers ==  129)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

# Plot bars with user-specified colors
m1 = ax.bar(x[5] - bar_width, penalty[keys[int(np.where(allowed_integers ==  130)[0])]], bar_width, color='white', hatch='-',  zorder=1, edgecolor=colors[25])
ax.errorbar(x[5] - bar_width, m1[0].get_height(), yerr = penalty_percentile[keys[int(np.where(allowed_integers ==  130)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

m2 = ax2.bar(x[5], HHI_capacity[keys[int(np.where(allowed_integers ==  130)[0])]], bar_width, color=colors[25], edgecolor=colors[25], zorder=1)
ax2.errorbar(x[5], m2[0].get_height(), yerr = HHI_capacity_percentile[keys[int(np.where(allowed_integers ==  130)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

m3 = ax3.bar(x[5] + bar_width, emissions[keys[int(np.where(allowed_integers ==  130)[0])]], bar_width, color='white', hatch='oo', edgecolor=colors[25])
ax3.errorbar(x[5] + bar_width, m3[0].get_height(), yerr = emissions_percentile[keys[int(np.where(allowed_integers ==  130)[0])],:].reshape(2,-1), fmt='none', ecolor='black', capsize=5)

# Labels and title
ax.set_title('LSTM with MLP in head')
ax2.set_ylabel('HHI', loc = 'center')
ax3.set_ylabel('Emissions [gTONCO2]', loc = 'bottom')
ax.set_xticks(x)
ax.set_xticklabels(run_names)
ax.set_yticks([])

ax.set_xticks(x)
ax.set_xticklabels(run_names)
ax3.spines['right'].set_position(('outward', 60))  


plt.tight_layout()
plt.show()
plt.savefig("training_metrics_EoM.pdf", format="pdf", bbox_inches="tight")