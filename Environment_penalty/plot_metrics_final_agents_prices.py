import os

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.ticker import MaxNLocator, FixedLocator

import pylab as pl

from matplotlib import rc
import matplotlib

matplotlib.rcParams['font.family'] = 'Nimbus Sans'



# Path to the parent directory containing folders

# data_path = '/dir/291224_results'

data_path = '/dir/020425_results_final_agents/'

run_codes = ['CM + CfD - 16 agents', 'EoM - 16 agents', 'CfD - 16 agents', 
             'CM + CfD - 32 agents', 'EoM - 32 agents', 'CfD - 32 agents', 
             'CM + CfD - 8 agents', 'EoM - 8 agents', 'CfD - 8 agents', 
             'CM - 16 agents', 'CM - 32 agents', 'CM - 8 agents']

checkpoint_runs = ['1_3_cm_storage_mask_lstm_173', '2_3_cm_storage_mask_lstm_174', '3_3_cm_storage_mask_lstm_175', 
                   '4_3_cm_storage_mask_lstm_176', '5_3_cm_storage_mask_lstm_177', '6_3_cm_storage_mask_lstm_178', 
                   '7_3_cm_storage_mask_lstm_179', '8_3_cm_storage_mask_lstm_180', '9_3_cm_storage_mask_lstm_181', 
                   '10_3_cm_storage_mask_lstm_182', '11_3_cm_storage_mask_lstm_183', '12_3_cm_storage_mask_lstm_184']

number_runs = 12

length_sim = 126

metric_prices = 6

prices_mean = np.zeros([number_runs, length_sim])

prices_median = np.zeros([number_runs, length_sim])

prices_percentile_75 = np.zeros([number_runs, length_sim])

prices_percentile_25 = np.zeros([number_runs, length_sim])

prices_percentile_75 = np.zeros([number_runs, length_sim])

prices_percentile_25 = np.zeros([number_runs, length_sim])


emissions_mean = np.zeros([number_runs, length_sim])

emissions_median = np.zeros([number_runs, length_sim])

emissions_max = np.zeros([number_runs, length_sim])

emissions_min = np.zeros([number_runs, length_sim])

emissions_percentile_75 = np.zeros([number_runs, length_sim])

emissions_percentile_25 = np.zeros([number_runs, length_sim])

n_tech = 7

capacity_2030_mean = np.zeros([number_runs, n_tech])

capacity_2030_median = np.zeros([number_runs, n_tech])

capacity_2030_max = np.zeros([number_runs, n_tech])

capacity_2030_min = np.zeros([number_runs, n_tech])

capacity_2030_percentile_75 = np.zeros([number_runs, n_tech])

capacity_2030_percentile_25 = np.zeros([number_runs, n_tech])


capacity_2040_mean = np.zeros([number_runs, n_tech])

capacity_2040_median = np.zeros([number_runs, n_tech])

capacity_2040_max = np.zeros([number_runs, n_tech])

capacity_2040_min = np.zeros([number_runs, n_tech])

capacity_2040_percentile_75 = np.zeros([number_runs, n_tech])

capacity_2040_percentile_25 = np.zeros([number_runs, n_tech])

# Read each CSV and append to matrices list
for i in range(number_runs):

    folder = f"{data_path}{checkpoint_runs[i]}"

    # Prices

    price_temp = pd.read_csv(os.path.join(folder, 'prices.csv')) # Read as numpy array
    price_temp = np.array(price_temp.iloc[:,1:])

    price_cut = price_temp[12:138,:]

    prices_mean[i,:] = np.mean(price_cut, axis=1)
    prices_median[i,:] = np.median(price_cut, axis=1)
    prices_percentile_75[i,:] = np.max(price_cut, axis=1)
    prices_percentile_25[i,:] = np.min(price_cut, axis=1)
    prices_percentile_75[i,:] = np.percentile(price_cut, 75, axis=1)
    prices_percentile_25[i,:] = np.percentile(price_cut, 25, axis=1)

    # Emissions

    emissions_temp = pd.read_csv(os.path.join(folder, 'CO2_emissions.csv')) # Read as numpy array
    emissions_temp = np.array(emissions_temp.iloc[:,1:])

    emissions_cut = emissions_temp[12:138,:]/(10**6)

    emissions_mean[i,:] = np.mean(emissions_cut, axis=1)
    emissions_median[i,:] = np.median(emissions_cut, axis=1)
    emissions_max[i,:] = np.max(emissions_cut, axis=1)
    emissions_min[i,:] = np.min(emissions_cut, axis=1)
    emissions_percentile_75[i,:] = np.percentile(emissions_cut, 75, axis=1)
    emissions_percentile_25[i,:] = np.percentile(emissions_cut, 25, axis=1)
    
    # Capacity 2030
    
    capacity_2030_temp = pd.read_csv(os.path.join(folder, 'capacity_2030.csv')) # Read as numpy array
    capacity_2030_temp = np.array(capacity_2030_temp.iloc[:,1:])

    capacity_2030_cut = capacity_2030_temp[:,:]/(10**3)

    capacity_2030_mean[i,:] = np.mean(capacity_2030_cut, axis=1)
    capacity_2030_median[i,:] = np.median(capacity_2030_cut, axis=1)
    capacity_2030_max[i,:] = np.max(capacity_2030_cut, axis=1)
    capacity_2030_min[i,:] = np.min(capacity_2030_cut, axis=1)
    capacity_2030_percentile_75[i,:] = np.percentile(capacity_2030_cut, 75, axis=1)
    capacity_2030_percentile_25[i,:] = np.percentile(capacity_2030_cut, 25, axis=1)
    
    # Capacity 2040
    
    capacity_2040_temp = pd.read_csv(os.path.join(folder, 'capacity_2040.csv')) # Read as numpy array
    capacity_2040_temp = np.array(capacity_2040_temp.iloc[:,1:])

    capacity_2040_cut = capacity_2040_temp[:,:]/(10**3)
    capacity_2040_mean[i,:] = np.mean(capacity_2040_cut, axis=1)
    capacity_2040_median[i,:] = np.median(capacity_2040_cut, axis=1)
    capacity_2040_max[i,:] = np.max(capacity_2040_cut, axis=1)
    capacity_2040_min[i,:] = np.min(capacity_2040_cut, axis=1)
    capacity_2040_percentile_75[i,:] = np.percentile(capacity_2040_cut, 75, axis=1)
    capacity_2040_percentile_25[i,:] = np.percentile(capacity_2040_cut, 25, axis=1)


# Create figure and subplots
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
plt.subplots_adjust(wspace=0, hspace=0)

colors = ['#156082', '#7F7F7F', '#00CAD2', '#007EC5', '#00A2E9', '#003F5C','#00E5FF', 
          '#156082', '#7F7F7F', '#00CAD2', '#007EC5', '#00A2E9', '#003F5C','#00E5FF',
          '#156082', '#7F7F7F', '#00CAD2', '#007EC5', '#00A2E9', '#003F5C','#00E5FF',
          '#156082', '#7F7F7F', '#00CAD2', '#007EC5', '#00A2E9', '#003F5C','#00E5FF',]

x_limit = 2040 + 5/6

x_axis = np.linspace(2020, x_limit, num=126)

# EoM - 8 agents

ax = axes[0,0]

run_codes = ['CM + CfD - 16 agents', 'EoM - 16 agents', 'CfD - 16 agents', 
             'CM + CfD - 32 agents', 'EoM - 32 agents', 'CfD - 32 agents', 
             'CM + CfD - 8 agents', 'EoM - 8 agents', 'CfD - 8 agents', 
             'CM - 16 agents', 'CM - 32 agents', 'CM - 8 agents']

# EoM - 8 agents
ax.plot(x_axis, prices_mean[7,:], label=run_codes[7] , color = colors[0])
ax.fill_between(x_axis, prices_percentile_25[7,:], prices_percentile_75[7,:], color = colors[0], alpha = 0.2 )

ax.xaxis.set_major_locator(plt.MaxNLocator(3))  # Reduce number of ticks


ax.set_xlim(2020, x_limit)
ax.set_ylim(40,200)
ax.yaxis.set_major_locator(plt.MaxNLocator(4))  # Reduce number of ticks
ax.set_title(run_codes[7])
ax.set_xticks([])


ax = axes[1,0]

# EoM - 16 agents
ax.plot(x_axis, prices_mean[1,:], label=run_codes[1] , color = colors[1])
ax.fill_between(x_axis, prices_percentile_25[1,:], prices_percentile_75[1,:], color = colors[1], alpha = 0.2 )

ax.xaxis.set_major_locator(plt.MaxNLocator(3))  # Reduce number of ticks
ax.yaxis.set_major_locator(plt.MaxNLocator(4))  # Reduce number of ticks

ax.set_xlim(2020, x_limit)
ax.set_ylim(40,200)
ax.set_title(run_codes[1])
ax.set_xticks([])

ax = axes[2,0]

# EoM - 32 agents
ax.plot(x_axis, prices_mean[4,:], label=run_codes[4] , color = colors[2])
ax.fill_between(x_axis, prices_percentile_25[4,:], prices_percentile_75[4,:], color = colors[2], alpha = 0.2 )

ax.xaxis.set_major_locator(plt.MaxNLocator(3))  # Reduce number of ticks
ax.yaxis.set_major_locator(plt.MaxNLocator(4))  # Reduce number of ticks

ax.set_xlim(2020, x_limit)
ax.set_ylim(40,200)
ax.set_title(run_codes[4])

## CM

ax = axes[0,1]

run_codes = ['CM + CfD - 16 agents', 'EoM - 16 agents', 'CfD - 16 agents', 
             'CM + CfD - 32 agents', 'EoM - 32 agents', 'CfD - 32 agents', 
             'CM + CfD - 8 agents', 'EoM - 8 agents', 'CfD - 8 agents', 
             'CM - 16 agents', 'CM - 32 agents', 'CM - 8 agents']

# CM - 8 agents
ax.plot(x_axis, prices_mean[11,:], label=run_codes[11] , color = colors[3])
ax.fill_between(x_axis, prices_percentile_25[11,:], prices_percentile_75[11,:], color = colors[3], alpha = 0.2 )
ax.xaxis.set_major_locator(plt.MaxNLocator(3))  # Reduce number of ticks
ax.yaxis.set_major_locator(plt.MaxNLocator(4))  # Reduce number of ticks
ax.set_title(run_codes[11])
ax.set_xlim(2020, x_limit)
ax.set_ylim(40,200)
ax.set_xticks([])
ax.set_yticks([])


ax = axes[1,1]
# CM - 16 agents
ax.plot(x_axis, prices_mean[9,:], label=run_codes[9] , color = colors[4])
ax.fill_between(x_axis, prices_percentile_25[9,:], prices_percentile_75[9,:], color = colors[4], alpha = 0.2 )
ax.xaxis.set_major_locator(plt.MaxNLocator(3))  # Reduce number of ticks
ax.yaxis.set_major_locator(plt.MaxNLocator(4))  # Reduce number of ticks
ax.set_title(run_codes[9])
ax.set_xlim(2020, x_limit)
ax.set_ylim(40,200)
ax.set_xticks([])
ax.set_yticks([])


ax = axes[2,1]

# CM - 32 agentss
ax.plot(x_axis, prices_mean[10,:], label=run_codes[10] , color = colors[5])
ax.fill_between(x_axis, prices_percentile_25[10,:], prices_percentile_75[10,:], color = colors[5], alpha = 0.2 )
ax.xaxis.set_major_locator(plt.MaxNLocator(3))  # Reduce number of ticks
ax.yaxis.set_major_locator(plt.MaxNLocator(4))  # Reduce number of ticks
ax.set_xlim(2020, x_limit)
ax.set_ylim(40,200)
ax.set_title(run_codes[10])
ax.set_yticks([])


## CfD

ax = axes[0,2]

run_codes = ['CM + CfD - 16 agents', 'EoM - 16 agents', 'CfD - 16 agents', 
             'CM + CfD - 32 agents', 'EoM - 32 agents', 'CfD - 32 agents', 
             'CM + CfD - 8 agents', 'EoM - 8 agents', 'CfD - 8 agents', 
             'CM - 16 agents', 'CM - 32 agents', 'CM - 8 agents']

# CfD - 8 agents
ax.plot(x_axis, prices_mean[8,:], label=run_codes[8] , color = colors[6])
ax.fill_between(x_axis, prices_percentile_25[8,:], prices_percentile_75[8,:], color = colors[6], alpha = 0.2 )
ax.xaxis.set_major_locator(plt.MaxNLocator(3))  # Reduce number of ticks
ax.yaxis.set_major_locator(plt.MaxNLocator(4))  # Reduce number of ticks

ax.set_xlim(2020, x_limit)
ax.set_ylim(40,200)
ax.set_title(run_codes[8])
ax.set_xticks([])
ax.set_yticks([])


# CfD - 16 agents
ax = axes[1,2]
ax.plot(x_axis, prices_mean[2,:], label=run_codes[2] , color = colors[7])
ax.fill_between(x_axis, prices_percentile_25[2,:], prices_percentile_75[2,:], color = colors[7], alpha = 0.2 )
ax.xaxis.set_major_locator(plt.MaxNLocator(3))  # Reduce number of ticks
ax.yaxis.set_major_locator(plt.MaxNLocator(4))  # Reduce number of ticks

ax.set_xlim(2020, x_limit)
ax.set_ylim(40,200)
ax.set_title(run_codes[2])
ax.set_xticks([])
ax.set_yticks([])

# CfD - 32 agents
ax = axes[2,2]
ax.plot(x_axis, prices_mean[5,:], label=run_codes[5] , color = colors[8])
ax.fill_between(x_axis, prices_percentile_25[5,:], prices_percentile_75[5,:], color = colors[8], alpha = 0.2 )
ax.xaxis.set_major_locator(plt.MaxNLocator(3))  # Reduce number of ticks
ax.yaxis.set_major_locator(plt.MaxNLocator(4))  # Reduce number of ticks

ax.set_xlim(2020, x_limit)
ax.set_ylim(40,200)
ax.set_title(run_codes[5])
ax.set_yticks([])


## CM + CfD

ax = axes[0,3]

run_codes = ['CM + CfD - 16 agents', 'EoM - 16 agents', 'CfD - 16 agents', 
             'CM + CfD - 32 agents', 'EoM - 32 agents', 'CfD - 32 agents', 
             'CM + CfD - 8 agents', 'EoM - 8 agents', 'CfD - 8 agents', 
             'CM - 16 agents', 'CM - 32 agents', 'CM - 8 agents']

# CM + CfD - 8 agents
ax.plot(x_axis, prices_mean[6,:], label=run_codes[6] , color = colors[9])
ax.fill_between(x_axis, prices_percentile_25[6,:], prices_percentile_75[6,:], color = colors[9], alpha = 0.2 )
ax.xaxis.set_major_locator(plt.MaxNLocator(3))  # Reduce number of ticks
ax.yaxis.set_major_locator(plt.MaxNLocator(4))  # Reduce number of ticks

ax.set_xlim(2020, x_limit)
ax.set_ylim(40,200)
ax.set_title(run_codes[6])
ax.set_xticks([])
ax.set_yticks([])


# CM + CfD - 16 agents
ax = axes[1,3]
ax.plot(x_axis, prices_mean[0,:], label=run_codes[0] , color = colors[10])
ax.fill_between(x_axis, prices_percentile_25[0,:], prices_percentile_75[0,:], color = colors[10], alpha = 0.2 )
ax.xaxis.set_major_locator(plt.MaxNLocator(3))  # Reduce number of ticks
ax.yaxis.set_major_locator(plt.MaxNLocator(4))  # Reduce number of ticks

ax.set_xlim(2020, x_limit)
ax.set_ylim(40,200)
ax.set_title(run_codes[0])
ax.set_xticks([])
ax.set_yticks([])


# CM + CfD - 32 agents
ax = axes[2,3]
ax.plot(x_axis, prices_mean[3,:], label=run_codes[3] , color = colors[11])
ax.fill_between(x_axis, prices_percentile_25[3,:], prices_percentile_75[3,:], color = colors[11], alpha = 0.2 )
ax.xaxis.set_major_locator(plt.MaxNLocator(3))  # Reduce number of ticks
ax.yaxis.set_major_locator(plt.MaxNLocator(4))  # Reduce number of ticks

ax.set_xlim(2020, x_limit)
ax.set_ylim(40,200)
ax.set_title(run_codes[3])
ax.set_yticks([])

fig.supxlabel('Year')
fig.supylabel('Total prices [€/MWh]')
plt.tight_layout()
plt.show()
plt.savefig("plot_metrics_final_agents_prices.pdf", format="pdf", bbox_inches="tight")

