import os

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import pylab as pl

from matplotlib import rc
import matplotlib

matplotlib.rcParams['font.family'] = 'Nimbus Sans'
import matplotlib.gridspec as gridspec

# Path to the parent directory containing folders

# data_path = '/dir/291224_results'

data_path = '/dir/020425_results_final_agents/'

run_codes = ['CM + CfD', 'EoM', 'CfD', 
             'CM + CfD', 'EoM', 'CfD', 
             'CM + CfD', 'EoM', 'CfD', 
             'CM', 'CM', 'CM']

checkpoint_runs = ['1_3_cm_storage_mask_lstm_173', '2_3_cm_storage_mask_lstm_174', '3_3_cm_storage_mask_lstm_175', 
                   '4_3_cm_storage_mask_lstm_176', '5_3_cm_storage_mask_lstm_177', '6_3_cm_storage_mask_lstm_178', 
                   '7_3_cm_storage_mask_lstm_179', '8_3_cm_storage_mask_lstm_180', '9_3_cm_storage_mask_lstm_181', 
                   '10_3_cm_storage_mask_lstm_182', '11_3_cm_storage_mask_lstm_183', '12_3_cm_storage_mask_lstm_184']

colors = ['#156082', '#7F7F7F', '#00CAD2', '#007EC5', '#00A2E9', '#003F5C','#00E5FF', '#00B89F', '#008CFF','#156082', '#00CAD2', '#007EC5', '#00A2E9', '#003F5C','#00E5FF', '#00B89F', '#008CFF','#156082', '#00CAD2', '#007EC5', '#00A2E9', '#003F5C','#00E5FF', '#00B89F', '#008CFF','#156082', '#00CAD2', '#007EC5', '#00A2E9', '#003F5C','#00E5FF', '#00B89F', '#008CFF']

capacity_alice = np.array([156, 7.8, 10.7, 0, 54.6, 50.3, 48.4])

number_runs = 12

length_sim = 126

metric_prices = 6

prices_mean = np.zeros([number_runs, length_sim])

prices_median = np.zeros([number_runs, length_sim])

prices_max = np.zeros([number_runs, length_sim])

prices_min = np.zeros([number_runs, length_sim])

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


capacity_merchant_2040 = np.zeros([number_runs, n_tech])

capacity_cm_2040 = np.zeros([number_runs, n_tech])

capacity_CfD_2040 = np.zeros([number_runs, n_tech])

total_capacity_2040 = np.zeros([number_runs, n_tech])

capacity_existing_2040 = np.zeros([number_runs, n_tech])

# Read each CSV and append to matrices list
for i in range(number_runs):

    folder = f"{data_path}{checkpoint_runs[i]}"

    # Prices

    price_temp = pd.read_csv(os.path.join(folder, 'prices.csv')) # Read as numpy array
    price_temp = np.array(price_temp.iloc[:,1:])

    price_cut = price_temp[12:138,:]

    prices_mean[i,:] = np.mean(price_cut, axis=1)
    prices_median[i,:] = np.median(price_cut, axis=1)
    prices_max[i,:] = np.max(price_cut, axis=1)
    prices_min[i,:] = np.min(price_cut, axis=1)
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

    total_capacity_2040 = capacity_2040_mean

    ## Capacity merchant - 2040

    capacity_merchant_tmp = pd.read_csv(os.path.join(folder, 'capacity_2040_merchant.csv')) # Read as numpy array
    capacity_merchant_tmp = np.array(capacity_merchant_tmp.iloc[:,1:])/1000

    capacity_merchant_2040[i,:] =  np.mean(capacity_merchant_tmp, axis = 1)
    
    ## Capacity CM - 2040

    capacity_cm_tmp = pd.read_csv(os.path.join(folder, 'capacity_2040_cm.csv')) # Read as numpy array
    capacity_cm_tmp = np.array(capacity_cm_tmp.iloc[:,1:])/1000

    capacity_cm_2040[i,:] = np.mean(capacity_cm_tmp, axis = 1) 

    ## Capacity CfD - 2040

    capacity_CfD_tmp = pd.read_csv(os.path.join(folder, 'capacity_2040_CfD.csv')) # Read as numpy array
    capacity_CfD_tmp = np.array(capacity_CfD_tmp.iloc[:,1:])/1000

    capacity_CfD_2040[i,:] = np.mean(capacity_CfD_tmp, axis = 1) 
    
    ## Total Capacity 2040 

    capacity_existing_2040[i,:] = capacity_2040_mean[i,:] - capacity_cm_2040[i,:] - capacity_merchant_2040[i,:] - capacity_CfD_2040[i,:]


total_capacity_2040 = np.clip(total_capacity_2040,0, None)

# Data
values = np.random.rand(13)  # Replace with actual values
labels = ["Bar 1"] + [f"G{i+1}-{j+1}" for i in range(3) for j in range(4)]


colors = ['#156082', '#7F7F7F', '#00CAD2', '#007EC5', '#00A2E9', '#003F5C','#00E5FF', '#00B89F', '#008CFF','#156082', '#00CAD2', '#007EC5', '#00A2E9', '#003F5C','#00E5FF', '#00B89F', '#008CFF','#156082', '#00CAD2', '#007EC5', '#00A2E9', '#003F5C','#00E5FF', '#00B89F', '#008CFF','#156082', '#00CAD2', '#007EC5', '#00A2E9', '#003F5C','#00E5FF', '#00B89F', '#008CFF']

# Bar positions
x_positions = [0] + list(np.arange(2, 6)) + list(np.arange(7, 11)) + list(np.arange(12, 16))

# Plot
fig = plt.figure(figsize=(17, 9))

gs = gridspec.GridSpec(12, 2, width_ratios=[1, 1], height_ratios=[1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1])

axes = []
## Solar 
int_graph = 0

ax = fig.add_subplot(gs[0:3, 0])
axes.append(ax)
ax = fig.add_subplot(gs[3:6, 0])
axes.append(ax)
ax = fig.add_subplot(gs[6:9, 0])
axes.append(ax)
ax = fig.add_subplot(gs[9:12, 0])
axes.append(ax)
ax = fig.add_subplot(gs[0:4, 1])
axes.append(ax)
ax = fig.add_subplot(gs[4:8, 1])
axes.append(ax)
ax = fig.add_subplot(gs[8:12:, 1])
axes.append(ax)

#for ax in fig.get_axes():
#    ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)

ax = axes[0]

run_codes = ['PYPSA','CM + CfD', 'EoM', 'CfD', 
             'CM + CfD', 'EoM', 'CfD', 
             'CM + CfD', 'EoM', 'CfD', 
             'CM', 'CM', 'CM']




run_order = np.array([7, 8, 11, 6,
             1, 2, 9, 0,
             4, 5, 10, 3], dtype=np.int8)


run_order_names = np.array([0, 8, 9, 12, 7,
             2, 3, 10, 1,
             5, 6, 11, 4], dtype=np.int8)


ax.bar(x_positions[0], capacity_alice[int_graph], color=colors[1], edgecolor=colors[1])  # First bar
ax.bar(x_positions[1:], capacity_existing_2040[run_order, int_graph], color=colors[0], edgecolor=colors[0], label="Existing")
ax.bar(x_positions[1:], capacity_merchant_2040[run_order, int_graph], bottom=capacity_existing_2040[run_order, int_graph], color=colors[2], edgecolor=colors[2], label="Merchant")
ax.bar(x_positions[1:], capacity_cm_2040[run_order, int_graph], bottom=capacity_existing_2040[run_order, int_graph] + capacity_merchant_2040[run_order, int_graph], color='white', hatch='xx', edgecolor=colors[3], label="CM")
m1 = ax.bar(x_positions[1:], capacity_CfD_2040[run_order, int_graph], bottom=capacity_existing_2040[run_order, int_graph] + capacity_merchant_2040[run_order, int_graph] + capacity_cm_2040[run_order, int_graph], color=colors[4], edgecolor=colors[4], label="CfD")

error_minus = total_capacity_2040[run_order, int_graph] - capacity_2040_min[run_order, int_graph]
error_max =  capacity_2040_max[run_order, int_graph] - total_capacity_2040[run_order, int_graph]
error = np.clip(np.array([error_minus, error_max]),0,None)
height = capacity_existing_2040[run_order, int_graph] + capacity_merchant_2040[run_order, int_graph] + capacity_cm_2040[run_order, int_graph] + capacity_CfD_2040[run_order, int_graph]
ax.errorbar(x_positions[1:], height, yerr = error, fmt='none', ecolor='black', capsize=0)

error_minus = total_capacity_2040[run_order, int_graph] - capacity_2040_percentile_25[run_order, int_graph]
error_max =  capacity_2040_percentile_75[run_order, int_graph] - total_capacity_2040[run_order, int_graph]
error = np.clip(np.array([error_minus, error_max]),0,None)
height = capacity_existing_2040[run_order, int_graph] + capacity_merchant_2040[run_order, int_graph] + capacity_cm_2040[run_order, int_graph] + capacity_CfD_2040[run_order, int_graph]
ax.errorbar(x_positions[1:], height, yerr = error, fmt='none', ecolor='black', capsize=5)

ax.set_xticks(x_positions)

ax.yaxis.set_major_locator(plt.MaxNLocator(4))
ax.set_xticks([])
separator_positions = [1, 6.1, 11.1] 
# Add vertical dotted lines between groups
for pos in separator_positions:
    ax.axvline(x=pos, color='grey', linestyle='--', linewidth=0.5)

ax.set_ylabel("[GW]")
ax.set_title("Solar PV")


## Onshore wind 
int_graph = 1

ax = axes[1]

run_codes = ['PYPSA','CM + CfD', 'EoM', 'CfD', 
             'CM + CfD', 'EoM', 'CfD', 
             'CM + CfD', 'EoM', 'CfD', 
             'CM', 'CM', 'CM']





run_order = np.array([7, 8, 11, 6,
             1, 2, 9, 0,
             4, 5, 10, 3], dtype=np.int8)


run_order_names = np.array([0, 8, 9, 12, 7,
             2, 3, 10, 1,
             5, 6, 11, 4], dtype=np.int8)


ax.bar(x_positions[0], capacity_alice[int_graph], color=colors[1], edgecolor=colors[1])  # First bar
ax.bar(x_positions[1:], capacity_existing_2040[run_order, int_graph], color=colors[0], edgecolor=colors[0], label="Existing")
ax.bar(x_positions[1:], capacity_merchant_2040[run_order, int_graph], bottom=capacity_existing_2040[run_order, int_graph], color=colors[2], edgecolor=colors[2], label="Merchant")
ax.bar(x_positions[1:], capacity_cm_2040[run_order, int_graph], bottom=capacity_existing_2040[run_order, int_graph] + capacity_merchant_2040[run_order, int_graph], color='white', hatch='xx', edgecolor=colors[3], label="CM")
m1 = ax.bar(x_positions[1:], capacity_CfD_2040[run_order, int_graph], bottom=capacity_existing_2040[run_order, int_graph] + capacity_merchant_2040[run_order, int_graph] + capacity_cm_2040[run_order, int_graph], color=colors[4], edgecolor=colors[4], label="CfD")

error_minus = total_capacity_2040[run_order, int_graph] - capacity_2040_min[run_order, int_graph]
error_max =  capacity_2040_max[run_order, int_graph] - total_capacity_2040[run_order, int_graph]
error = np.clip(np.array([error_minus, error_max]),0,None)
height = capacity_existing_2040[run_order, int_graph] + capacity_merchant_2040[run_order, int_graph] + capacity_cm_2040[run_order, int_graph] + capacity_CfD_2040[run_order, int_graph]
ax.errorbar(x_positions[1:], height, yerr = error, fmt='none', ecolor='black', capsize=0)

error_minus = total_capacity_2040[run_order, int_graph] - capacity_2040_percentile_25[run_order, int_graph]
error_max =  capacity_2040_percentile_75[run_order, int_graph] - total_capacity_2040[run_order, int_graph]
error = np.clip(np.array([error_minus, error_max]),0,None)
height = capacity_existing_2040[run_order, int_graph] + capacity_merchant_2040[run_order, int_graph] + capacity_cm_2040[run_order, int_graph] + capacity_CfD_2040[run_order, int_graph]
ax.errorbar(x_positions[1:], height, yerr = error, fmt='none', ecolor='black', capsize=5)


ax.set_xticks(x_positions)

ax.yaxis.set_major_locator(plt.MaxNLocator(4))
ax.set_xticks([])
separator_positions = [1, 6.1, 11.1] 
# Add vertical dotted lines between groups
for pos in separator_positions:
    ax.axvline(x=pos, color='grey', linestyle='--', linewidth=0.5)
ax.set_ylabel("[GW]")
ax.set_title("Onshore wind")


## Offshore wind 
int_graph = 2

ax = axes[2]

run_codes = ['PYPSA','CM + CfD', 'EoM', 'CfD', 
             'CM + CfD', 'EoM', 'CfD', 
             'CM + CfD', 'EoM', 'CfD', 
             'CM', 'CM', 'CM']





run_order = np.array([7, 8, 11, 6,
             1, 2, 9, 0,
             4, 5, 10, 3], dtype=np.int8)


run_order_names = np.array([0, 8, 9, 12, 7,
             2, 3, 10, 1,
             5, 6, 11, 4], dtype=np.int8)


ax.bar(x_positions[0], capacity_alice[int_graph], color=colors[1], edgecolor=colors[1])  # First bar
ax.bar(x_positions[1:], capacity_existing_2040[run_order, int_graph], color=colors[0], edgecolor=colors[0], label="Existing")
ax.bar(x_positions[1:], capacity_merchant_2040[run_order, int_graph], bottom=capacity_existing_2040[run_order, int_graph], color=colors[2], edgecolor=colors[2], label="Merchant")
ax.bar(x_positions[1:], capacity_cm_2040[run_order, int_graph], bottom=capacity_existing_2040[run_order, int_graph] + capacity_merchant_2040[run_order, int_graph], color='white', hatch='xx', edgecolor=colors[3], label="CM")
m1 = ax.bar(x_positions[1:], capacity_CfD_2040[run_order, int_graph], bottom=capacity_existing_2040[run_order, int_graph] + capacity_merchant_2040[run_order, int_graph] + capacity_cm_2040[run_order, int_graph], color=colors[4], edgecolor=colors[4], label="CfD")

error_minus = total_capacity_2040[run_order, int_graph] - capacity_2040_min[run_order, int_graph]
error_max =  capacity_2040_max[run_order, int_graph] - total_capacity_2040[run_order, int_graph]
error = np.clip(np.array([error_minus, error_max]),0,None)

height = capacity_existing_2040[run_order, int_graph] + capacity_merchant_2040[run_order, int_graph] + capacity_cm_2040[run_order, int_graph] + capacity_CfD_2040[run_order, int_graph]
ax.errorbar(x_positions[1:], height, yerr = error, fmt='none', ecolor='black', capsize=0)

error_minus = total_capacity_2040[run_order, int_graph] - capacity_2040_percentile_25[run_order, int_graph]
error_max =  capacity_2040_percentile_75[run_order, int_graph] - total_capacity_2040[run_order, int_graph]
error = np.clip(np.array([error_minus, error_max]),0,None)
height = capacity_existing_2040[run_order, int_graph] + capacity_merchant_2040[run_order, int_graph] + capacity_cm_2040[run_order, int_graph] + capacity_CfD_2040[run_order, int_graph]
ax.errorbar(x_positions[1:], height, yerr = error, fmt='none', ecolor='black', capsize=5)


ax.set_xticks(x_positions)

ax.yaxis.set_major_locator(plt.MaxNLocator(4))
ax.set_xticks([])
separator_positions = [1, 6.1, 11.1] 
# Add vertical dotted lines between groups
for pos in separator_positions:
    ax.axvline(x=pos, color='grey', linestyle='--', linewidth=0.5)

ax.set_ylabel("[GW]")
ax.set_title("Offshore wind")

## Coal
int_graph = 3

ax = axes[4]

run_codes = ['PYPSA','CM + CfD', 'EoM', 'CfD', 
             'CM + CfD', 'EoM', 'CfD', 
             'CM + CfD', 'EoM', 'CfD', 
             'CM', 'CM', 'CM']





run_order = np.array([7, 8, 11, 6,
             1, 2, 9, 0,
             4, 5, 10, 3], dtype=np.int8)


run_order_names = np.array([0, 8, 9, 12, 7,
             2, 3, 10, 1,
             5, 6, 11, 4], dtype=np.int8)


ax.bar(x_positions[0], capacity_alice[int_graph], color=colors[1], edgecolor=colors[1])  # First bar
ax.bar(x_positions[1:], capacity_existing_2040[run_order, int_graph], color=colors[0], edgecolor=colors[0], label="Existing")
ax.bar(x_positions[1:], capacity_merchant_2040[run_order, int_graph], bottom=capacity_existing_2040[run_order, int_graph], color=colors[2], edgecolor=colors[2], label="Merchant")
ax.bar(x_positions[1:], capacity_cm_2040[run_order, int_graph], bottom=capacity_existing_2040[run_order, int_graph] + capacity_merchant_2040[run_order, int_graph], color=colors[2], edgecolor=colors[2], label="CM")
m1 = ax.bar(x_positions[1:], capacity_CfD_2040[run_order, int_graph], bottom=capacity_existing_2040[run_order, int_graph] + capacity_merchant_2040[run_order, int_graph] + capacity_cm_2040[run_order, int_graph], color=colors[4], edgecolor=colors[4], label="CfD")

error_minus = total_capacity_2040[run_order, int_graph] - capacity_2040_min[run_order, int_graph]
error_max =  capacity_2040_max[run_order, int_graph] - total_capacity_2040[run_order, int_graph]
error = np.clip(np.array([error_minus, error_max]),0,None)
height = capacity_existing_2040[run_order, int_graph] + capacity_merchant_2040[run_order, int_graph] + capacity_cm_2040[run_order, int_graph] + capacity_CfD_2040[run_order, int_graph]
ax.errorbar(x_positions[1:], height, yerr = error, fmt='none', ecolor='black', capsize=0)

error_minus = total_capacity_2040[run_order, int_graph] - capacity_2040_percentile_25[run_order, int_graph]
error_max =  capacity_2040_percentile_75[run_order, int_graph] - total_capacity_2040[run_order, int_graph]
error = np.clip(np.array([error_minus, error_max]),0,None)
height = capacity_existing_2040[run_order, int_graph] + capacity_merchant_2040[run_order, int_graph] + capacity_cm_2040[run_order, int_graph] + capacity_CfD_2040[run_order, int_graph]
ax.errorbar(x_positions[1:], height, yerr = error, fmt='none', ecolor='black', capsize=5)


ax.set_xticks(x_positions)
ax.yaxis.set_major_locator(plt.MaxNLocator(4))
ax.set_xticks([])
separator_positions = [1, 6.1, 11.1] 
# Add vertical dotted lines between groups
for pos in separator_positions:
    ax.axvline(x=pos, color='grey', linestyle='--', linewidth=0.5)

ax.set_ylabel("[GW]")
ax.set_title("Coal")
ax.set_ylim(0,1)

## OCGT
int_graph = 4

ax = axes[5]

run_codes = ['PYPSA','CM + CfD', 'EoM', 'CfD', 
             'CM + CfD', 'EoM', 'CfD', 
             'CM + CfD', 'EoM', 'CfD', 
             'CM', 'CM', 'CM']





run_order = np.array([7, 8, 11, 6,
             1, 2, 9, 0,
             4, 5, 10, 3], dtype=np.int8)


run_order_names = np.array([0, 8, 9, 12, 7,
             2, 3, 10, 1,
             5, 6, 11, 4], dtype=np.int8)


ax.bar(x_positions[0], capacity_alice[int_graph], color=colors[1], edgecolor=colors[1])  # First bar
ax.bar(x_positions[1:], capacity_existing_2040[run_order, int_graph], color=colors[0], edgecolor=colors[0], label="Existing")
ax.bar(x_positions[1:], capacity_merchant_2040[run_order, int_graph], bottom=capacity_existing_2040[run_order, int_graph], color=colors[2], edgecolor=colors[2], label="Merchant")
ax.bar(x_positions[1:], capacity_cm_2040[run_order, int_graph], bottom=capacity_existing_2040[run_order, int_graph] + capacity_merchant_2040[run_order, int_graph], color='white', hatch='xx', edgecolor=colors[3], label="CM")
m1 = ax.bar(x_positions[1:], capacity_CfD_2040[run_order, int_graph], bottom=capacity_existing_2040[run_order, int_graph] + capacity_merchant_2040[run_order, int_graph] + capacity_cm_2040[run_order, int_graph], color=colors[4], edgecolor=colors[4], label="CfD")

error_minus = total_capacity_2040[run_order, int_graph] - capacity_2040_min[run_order, int_graph]
error_max =  capacity_2040_max[run_order, int_graph] - total_capacity_2040[run_order, int_graph]
error = np.clip(np.array([error_minus, error_max]),0,None)
height = capacity_existing_2040[run_order, int_graph] + capacity_merchant_2040[run_order, int_graph] + capacity_cm_2040[run_order, int_graph] + capacity_CfD_2040[run_order, int_graph]
ax.errorbar(x_positions[1:], height, yerr = error, fmt='none', ecolor='black', capsize=0)

error_minus = total_capacity_2040[run_order, int_graph] - capacity_2040_percentile_25[run_order, int_graph]
error_max =  capacity_2040_percentile_75[run_order, int_graph] - total_capacity_2040[run_order, int_graph]
error = np.clip(np.array([error_minus, error_max]),0,None)
height = capacity_existing_2040[run_order, int_graph] + capacity_merchant_2040[run_order, int_graph] + capacity_cm_2040[run_order, int_graph] + capacity_CfD_2040[run_order, int_graph]
ax.errorbar(x_positions[1:], height, yerr = error, fmt='none', ecolor='black', capsize=5)


ax.set_xticks(x_positions)
ax.yaxis.set_major_locator(plt.MaxNLocator(4))
ax.set_xticks([])
separator_positions = [1, 6.1, 11.1] 
# Add vertical dotted lines between groups
for pos in separator_positions:
    ax.axvline(x=pos, color='grey', linestyle='--', linewidth=0.5)

ax.set_ylabel("[GW]")
ax.set_title("OCGT")

## CCGT
int_graph = 5

ax = axes[6]

run_codes = ['PYPSA','CM + CfD', 'EoM', 'CfD', 
             'CM + CfD', 'EoM', 'CfD', 
             'CM + CfD', 'EoM', 'CfD', 
             'CM', 'CM', 'CM']

run_order = np.array([7, 8, 11, 6,
             1, 2, 9, 0,
             4, 5, 10, 3], dtype=np.int8)


run_order_names = np.array([0, 8, 9, 12, 7,
             2, 3, 10, 1,
             5, 6, 11, 4], dtype=np.int8)


ax.bar(x_positions[0], capacity_alice[int_graph], color=colors[1], edgecolor=colors[1])  # First bar
ax.bar(x_positions[1:], capacity_existing_2040[run_order, int_graph], color=colors[0], edgecolor=colors[0], label="Existing")
ax.bar(x_positions[1:], capacity_merchant_2040[run_order, int_graph], bottom=capacity_existing_2040[run_order, int_graph], color=colors[2], edgecolor=colors[2], label="Merchant")
ax.bar(x_positions[1:], capacity_cm_2040[run_order, int_graph], bottom=capacity_existing_2040[run_order, int_graph] + capacity_merchant_2040[run_order, int_graph], color='white', hatch='xx', edgecolor=colors[3], label="CM")
m1 = ax.bar(x_positions[1:], capacity_CfD_2040[run_order, int_graph], bottom=capacity_existing_2040[run_order, int_graph] + capacity_merchant_2040[run_order, int_graph] + capacity_cm_2040[run_order, int_graph], color=colors[4], edgecolor=colors[4], label="CfD")

error_minus = (total_capacity_2040[run_order, int_graph] - capacity_2040_min[run_order, int_graph]) 
print(error_minus)
error_max =  capacity_2040_max[run_order, int_graph] - total_capacity_2040[run_order, int_graph]
print(error_max)
error = np.clip(np.array([error_minus, error_max]),0,None)
height = capacity_existing_2040[run_order, int_graph] + capacity_merchant_2040[run_order, int_graph] + capacity_cm_2040[run_order, int_graph] + capacity_CfD_2040[run_order, int_graph]
ax.errorbar(x_positions[1:], height, yerr = error, fmt='none', ecolor='black', capsize=0)

error_minus = total_capacity_2040[run_order, int_graph] - capacity_2040_percentile_25[run_order, int_graph]
error_max =  capacity_2040_percentile_75[run_order, int_graph] - total_capacity_2040[run_order, int_graph]
error = np.clip(np.array([error_minus, error_max]),0,None)
height = capacity_existing_2040[run_order, int_graph] + capacity_merchant_2040[run_order, int_graph] + capacity_cm_2040[run_order, int_graph] + capacity_CfD_2040[run_order, int_graph]
ax.errorbar(x_positions[1:], height, yerr = error, fmt='none', ecolor='black', capsize=5)


ax.set_xticks(x_positions)
ax.set_xticklabels(np.array(run_codes)[run_order_names], rotation=45, ha="right")

ax.yaxis.set_major_locator(plt.MaxNLocator(4))
group_positions = [3.4, 8.6, 13.2]
group_labels = ["8 Agents", "16 Agents", "32 Agents"]
for i, group_label in enumerate(group_labels):
    ax.text(group_positions[i], -0.3, group_label, ha='center', va='top', fontsize=12, fontweight='bold', transform=ax.get_xaxis_transform())

separator_positions = [1, 6.1, 11.1] 
# Add vertical dotted lines between groups
for pos in separator_positions:
    ax.axvline(x=pos, color='grey', linestyle='--', linewidth=0.5)
#ax.legend(loc = "upper right")
ax.set_ylabel("[GW]")
ax.set_title("CCGT")

## Batteries
int_graph = 6

ax = axes[3]

run_codes = ['PYPSA','CM + CfD', 'EoM', 'CfD', 
             'CM + CfD', 'EoM', 'CfD', 
             'CM + CfD', 'EoM', 'CfD', 
             'CM', 'CM', 'CM']





run_order = np.array([7, 8, 11, 6,
             1, 2, 9, 0,
             4, 5, 10, 3], dtype=np.int8)


run_order_names = np.array([0, 8, 9, 12, 7,
             2, 3, 10, 1,
             5, 6, 11, 4], dtype=np.int8)


ax.bar(x_positions[0], capacity_alice[int_graph], color=colors[1], edgecolor=colors[1])  # First bar
ax.bar(x_positions[1:], capacity_existing_2040[run_order, int_graph], color=colors[0], edgecolor=colors[0], label="Existing")
ax.bar(x_positions[1:], capacity_merchant_2040[run_order, int_graph], bottom=capacity_existing_2040[run_order, int_graph], color=colors[2], edgecolor=colors[2], label="Merchant")
ax.bar(x_positions[1:], capacity_cm_2040[run_order, int_graph], bottom=capacity_existing_2040[run_order, int_graph] + capacity_merchant_2040[run_order, int_graph], color='white', hatch='xx', edgecolor=colors[3], label="CM")
m1 = ax.bar(x_positions[1:], capacity_CfD_2040[run_order, int_graph], bottom=capacity_existing_2040[run_order, int_graph] + capacity_merchant_2040[run_order, int_graph] + capacity_cm_2040[run_order, int_graph], color=colors[4], edgecolor=colors[4], label="CfD")

error_minus = total_capacity_2040[run_order, int_graph] - capacity_2040_min[run_order, int_graph]
error_max =  capacity_2040_max[run_order, int_graph] - total_capacity_2040[run_order, int_graph]
error = np.clip(np.array([error_minus, error_max]),0,None)
height = capacity_existing_2040[run_order, int_graph] + capacity_merchant_2040[run_order, int_graph] + capacity_cm_2040[run_order, int_graph] + capacity_CfD_2040[run_order, int_graph]
ax.errorbar(x_positions[1:], height, yerr = error, fmt='none', ecolor='black', capsize=0)

error_minus = total_capacity_2040[run_order, int_graph] - capacity_2040_percentile_25[run_order, int_graph]
error_max =  capacity_2040_percentile_75[run_order, int_graph] - total_capacity_2040[run_order, int_graph]
error = np.clip(np.array([error_minus, error_max]),0,None)
height = capacity_existing_2040[run_order, int_graph] + capacity_merchant_2040[run_order, int_graph] + capacity_cm_2040[run_order, int_graph] + capacity_CfD_2040[run_order, int_graph]
ax.errorbar(x_positions[1:], height, yerr = error, fmt='none', ecolor='black', capsize=5)


ax.set_xticks(x_positions)
ax.legend()

ax.yaxis.set_major_locator(plt.MaxNLocator(4))
ax.set_xticklabels(np.array(run_codes)[run_order_names], rotation=45, ha="right")
ax.set_ylabel("[GW]")
ax.set_title("Short-term Storage - Batteries")


group_labels = ["8 Agents", "16 Agents", "32 Agents"]
for i, group_label in enumerate(group_labels):
    ax.text(group_positions[i], -0.4, group_label, ha='center', va='top', fontsize=12, fontweight='bold', transform=ax.get_xaxis_transform())

separator_positions = [1, 6.1, 11.1] 
# Add vertical dotted lines between groups
for pos in separator_positions:
    ax.axvline(x=pos, color='grey', linestyle='--', linewidth=0.5)

plt.tight_layout()  
plt.savefig("plot_bars_capacities_2040.pdf", format="pdf", bbox_inches="tight")

