import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib import rc
import matplotlib

matplotlib.rcParams['font.family'] = 'Nimbus Sans'

# Create figure and subplots
fig, axes = plt.subplots(2, 3, figsize=(12, 6), subplot_kw=dict(polar=True))
plt.rcParams.update({'font.size': 10})
#plt.subplots_adjust(wspace=0, hspace=0)


import matplotlib.colors as mcolors
import numpy as np

labels_all = ["A-Penalty", "B-Penalty", "A-HHI", "B-HHI","A-Rank", "B-Rank"]

labels_not_center = ["", "B-Penalty", "A-HHI", "","A-Rank", "B-Rank"]

labels_center = ["A-Penalty", "", "", "B-HHI","", ""]

labels_empty = ["", "", "", "", "", "","", ""]

num_vars = len(labels_all)

colors = ['#156082', '#7F7F7F', '#00CAD2', '#007EC5', '#00A2E9', '#003F5C','#00E5FF', '#00B89F', '#008CFF','#156082', '#00CAD2', '#007EC5', '#00A2E9', '#003F5C','#00E5FF', '#00B89F', '#008CFF','#156082', '#00CAD2', '#007EC5', '#00A2E9', '#003F5C','#00E5FF', '#00B89F', '#008CFF','#156082', '#00CAD2', '#007EC5', '#00A2E9', '#003F5C','#00E5FF', '#00B89F', '#008CFF']

## Clipping factor

allowed_integers_EoM = np.array([66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 84, 85, 86, 87, 88, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130])

keys = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]) 

rank= pd.read_csv('030325_analysis/030325 Rank.csv')  # Read as numpy array
rank = np.array(rank.iloc[:,:])

print(rank)

rank_EoM = np.clip((rank[:,1] - np.min(rank[:,1]))/np.median(rank[:,1] - np.min(rank[:,1])), 0, 2) + 1
rank_CM_CfD = np.clip((rank[:,2] - np.min(rank[:,2]))/np.median(rank[:,2] - np.min(rank[:,2])), 0, 2) + 1

penalty_EoM = pd.read_csv('030325_analysis/total_penalty.csv')  # Read as numpy array
penalty_EoM = np.array(penalty_EoM.iloc[:,1])
penalty_EoM = np.clip((penalty_EoM - np.min(penalty_EoM))/np.median(penalty_EoM - np.min(penalty_EoM)), 0, 2) + 1

HHI_capacity_EoM = pd.read_csv('030325_analysis/HHI_capacity_2040.csv')  # Read as numpy array
HHI_capacity_EoM = np.array(HHI_capacity_EoM.iloc[:,1])
HHI_capacity_EoM = np.clip((HHI_capacity_EoM - np.min(HHI_capacity_EoM))/np.median(HHI_capacity_EoM - np.min(HHI_capacity_EoM)), 0, 2) + 1

allowed_integers_CM_CfD = np.array([46, 47, 48, 52, 53, 56, 57, 60, 61, 64, 79, 80, 81, 82, 83, 89, 91, 93, 95, 97, 99, 101, 103, 105, 107, 109, 111]) 

penalty_CM_CfD = pd.read_csv('040125_analysis/total_penalty.csv')  # Read as numpy array
penalty_CM_CfD = np.array(penalty_CM_CfD.iloc[:,1])
penalty_CM_CfD = np.clip((penalty_CM_CfD - np.min(penalty_CM_CfD))/np.median(penalty_CM_CfD - np.min(penalty_CM_CfD)), 0, 2) + 1

HHI_capacity_CM_CfD = pd.read_csv('040125_analysis/HHI_capacity_2040.csv')  # Read as numpy array
HHI_capacity_CM_CfD = np.array(HHI_capacity_CM_CfD.iloc[:,1])
HHI_capacity_CM_CfD = np.clip((HHI_capacity_CM_CfD - np.min(HHI_capacity_CM_CfD))/np.median(HHI_capacity_CM_CfD - np.min(HHI_capacity_CM_CfD)), 0, 2) + 1

# Compute angles for each characteristic
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

# Close the plot

angles += [angles[0]]

labels_empty = [
    "", "","","","",""
]

linestyles = ['-','--','-','--','-','--']

# Loop through subplots and plot the same radar chart in both

## Clipping factor

row = 0
column = 0

ax = axes[row,column]

# 87,67,71,72

# AM.1, AM.2, AM.3, AM.4

# 82, 47, 56, 57

# BM.1, BM.2, BM.3, BM.4

integers = np.array([[87, 67, 71, 72], [82, 47, 56, 57]])
colors_index = np.array([0, 1, 2, 3])

run_names = ["M.1", "M.2", "M.3", "M.4"]

for i in range(4):

    v = [penalty_EoM[keys[int(np.where(allowed_integers_EoM == integers[0,i])[0])]],   penalty_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == integers[1,i])[0])]],
        HHI_capacity_EoM[keys[int(np.where(allowed_integers_EoM == integers[0,i])[0])]],   HHI_capacity_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == integers[1,i])[0])]],
        rank_EoM[keys[int(np.where(allowed_integers_EoM == integers[0,i])[0])]],   rank_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == integers[1,i])[0])]],
        penalty_EoM[keys[int(np.where(allowed_integers_EoM == integers[0,i])[0])]]]

    ax.plot(angles, v, label = run_names[i], linewidth=2, zorder = 2, color = colors[colors_index[i]], linestyle = linestyles[i])
    # Add error bars as vertical lines
    # Format the radar chart
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels_not_center, fontsize=10, backgroundcolor="white", zorder=10)
    ax.set_yticks([1, 2, 3])  # Choose which tick values to display
    ax.set_yticklabels(["1.0", "2.0", "3.0"], fontsize=8)
    ax.set_ylim(0, 3)
    for angle, label in zip(angles[:-1], labels_center):
        ax.text(angle, 3.8, label, ha='center', va='center', fontsize=10, zorder=10)
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.2))
    #ax.legend(loc="best")
    ax.set_title('Clipping Factor')

## Batch Size

row = 0
column = 1

ax = axes[row,column]

# 70,69,67,75,85,86

# AM.5, AM.6, AM.2, AM.7, AM.8, AM.9

# BM.1, BM.2, BM.3, BM.4

# 53, 52, 47, 64, 80, 81,

# BM.5, BM.6, BM.2, BM.7, BM.8, BM.9

integers = np.array([[70,69,67,75,85,86], [53, 52, 47, 64, 80, 81]])
colors_index = np.array([4, 5, 1, 6, 7, 8])

run_names = ["M.5", "M.6", "M.2", "M.7", "M.8", "M.9"]

for i in range(6):

    labels_all = ["A-Penalty", "B-Penalty", "A-HHI", "B-HHI","A-Rank", "B-Rank"]

    v = [penalty_EoM[keys[int(np.where(allowed_integers_EoM == integers[0,i])[0])]],   penalty_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == integers[1,i])[0])]],
        HHI_capacity_EoM[keys[int(np.where(allowed_integers_EoM == integers[0,i])[0])]],   HHI_capacity_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == integers[1,i])[0])]],
        rank_EoM[keys[int(np.where(allowed_integers_EoM == integers[0,i])[0])]],   rank_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == integers[1,i])[0])]],
        penalty_EoM[keys[int(np.where(allowed_integers_EoM == integers[0,i])[0])]]]

    if i == 4:
        print(v)

    ax.plot(angles, v, label = run_names[i], linewidth=2, zorder = 2, color = colors[colors_index[i]], linestyle = linestyles[i])
    # Add error bars as vertical lines
    # Format the radar chart
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels_not_center, fontsize=10, backgroundcolor="white", zorder=10)
    ax.set_yticks([1, 2, 3])  # Choose which tick values to display
    ax.set_yticklabels(["1.0", "2.0", "3.0"], fontsize=8)
    ax.set_ylim(0, 3)
    for angle, label in zip(angles[:-1], labels_center):
        ax.text(angle, 3.8, label, ha='center', va='center', fontsize=10, zorder=10)
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.2))
    #ax.legend(loc="best")
    ax.set_title('Batch Size')

## Entropy

row = 0
column = 2

ax = axes[row,column]

# AM.10, AM.2, AM.11
# 66,67,68
# BM.10, BM.2, BM.11
# 46, 47, 48

integers = np.array([[66,67,68], [46, 47, 48]])
colors_index = np.array([9, 1, 10])

run_names = ["M.10", "M.2", "M.11"]

for i in range(3):

    labels_all = ["A-Penalty", "B-Penalty", "A-HHI", "B-HHI","A-Rank", "B-Rank"]

    v = [penalty_EoM[keys[int(np.where(allowed_integers_EoM == integers[0,i])[0])]],   penalty_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == integers[1,i])[0])]],
        HHI_capacity_EoM[keys[int(np.where(allowed_integers_EoM == integers[0,i])[0])]],   HHI_capacity_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == integers[1,i])[0])]],
        rank_EoM[keys[int(np.where(allowed_integers_EoM == integers[0,i])[0])]],   rank_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == integers[1,i])[0])]],
        penalty_EoM[keys[int(np.where(allowed_integers_EoM == integers[0,i])[0])]]]

    if i == 4:
        print(v)

    ax.plot(angles, v, label = run_names[i], linewidth=2, zorder = 2, color = colors[colors_index[i]], linestyle = linestyles[i])
    # Add error bars as vertical lines
    # Format the radar chart
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels_not_center, fontsize=10, backgroundcolor="white", zorder=10)
    ax.set_yticks([1, 2, 3])  # Choose which tick values to display
    ax.set_yticklabels(["1.0", "2.0", "3.0"], fontsize=8)
    ax.set_ylim(0, 3)
    for angle, label in zip(angles[:-1], labels_center):
        ax.text(angle, 3.8, label, ha='center', va='center', fontsize=10, zorder=10)
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.2))
    #ax.legend(loc="best")
    ax.set_title('Entropy')


## MLP Configuration

row = 1
column = 0

ax = axes[row,column]

# AM.12, AM.2, AM.13, AM.14
# 73,67,88,74

# BM.12, BM.2, BM.13, BM.14
# 60, 47, 83, 61

integers = np.array([[73, 67, 88, 74], [60, 47, 83, 61]])
colors_index = np.array([11, 1, 12, 13])
run_names = ["M.12", "M.2", "M.13", "M.14"]

for i in range(4):

    v = [penalty_EoM[keys[int(np.where(allowed_integers_EoM == integers[0,i])[0])]],   penalty_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == integers[1,i])[0])]],
        HHI_capacity_EoM[keys[int(np.where(allowed_integers_EoM == integers[0,i])[0])]],   HHI_capacity_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == integers[1,i])[0])]],
        rank_EoM[keys[int(np.where(allowed_integers_EoM == integers[0,i])[0])]],   rank_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == integers[1,i])[0])]],
        penalty_EoM[keys[int(np.where(allowed_integers_EoM == integers[0,i])[0])]]]

    ax.plot(angles, v, label = run_names[i], linewidth=2, zorder = 2, color = colors[colors_index[i]], linestyle = linestyles[i])
    # Add error bars as vertical lines
    # Format the radar chart
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels_not_center, fontsize=10, backgroundcolor="white", zorder=10)
    ax.set_yticks([1, 2, 3])  # Choose which tick values to display
    ax.set_yticklabels(["1.0", "2.0", "3.0"], fontsize=8)
    ax.set_ylim(0, 3)
    for angle, label in zip(angles[:-1], labels_center):
        ax.text(angle, 3.8, label, ha='center', va='center', fontsize=10, zorder=10)
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.2))
    #ax.legend(loc="best")
    ax.set_title('MLP Configuration')


## LSTM using MLP in tail

row = 1
column = 1

ax = axes[row,column]

# AL.1, AL.2, AL.3, AL.4, AL.5, AL.6
# 120, 119, 121, 122, 123, 124

# BL.1, BL.2, BL.3, BL.4, BL.5, BL.6
# 91, 89, 93, 95, 97, 99


integers = np.array([[120, 119, 121, 122, 123, 124], [91, 89, 93, 95, 97, 99]])
colors_index = np.array([14, 15, 16, 17, 18, 19])

run_names = ["L.1", "L.2", "L.3", "L.4", "L.5", "L.6"]

for i in range(6):

    labels_all = ["A-Penalty", "B-Penalty", "A-HHI", "B-HHI","A-Rank", "B-Rank"]

    v = [penalty_EoM[keys[int(np.where(allowed_integers_EoM == integers[0,i])[0])]],   penalty_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == integers[1,i])[0])]],
        HHI_capacity_EoM[keys[int(np.where(allowed_integers_EoM == integers[0,i])[0])]],   HHI_capacity_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == integers[1,i])[0])]],
        rank_EoM[keys[int(np.where(allowed_integers_EoM == integers[0,i])[0])]],   rank_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == integers[1,i])[0])]],
        penalty_EoM[keys[int(np.where(allowed_integers_EoM == integers[0,i])[0])]]]

    if i == 4:
        print(v)

    ax.plot(angles, v, label = run_names[i], linewidth=2, zorder = 2, color = colors[colors_index[i]], linestyle = linestyles[i])
    # Add error bars as vertical lines
    # Format the radar chart
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels_not_center, fontsize=10, backgroundcolor="white", zorder=10)
    ax.set_yticks([1, 2, 3])  # Choose which tick values to display
    ax.set_yticklabels(["1.0", "2.0", "3.0"], fontsize=8)
    ax.set_ylim(0, 3)
    for angle, label in zip(angles[:-1], labels_center):
        ax.text(angle, 3.8, label, ha='center', va='center', fontsize=10, zorder=10)
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.2))
    #ax.legend(loc="best")
    ax.set_title('LSTM with MLP in tail')

## LSTM using MLP in HEAD

row = 1
column = 2

ax = axes[row,column]

# AL.7, AL.8, AL.9, AL.10, AL.11, AL.12
# 126, 125, 127, 128, 129, 130

# BL.7, BL.8, BL.9, BL.10, BL.11, BL.12
# 103, 101, 105, 107, 109, 111


integers = np.array([[126, 125, 127, 128, 129, 130], [103, 101, 105, 107, 109, 111]])
colors_index = np.array([20, 21, 22, 23, 24, 25])

run_names = ["L.7", "L.8", "L.9", "L.10", "L.11", "L.12"]

for i in range(6):

    labels_all = ["A-Penalty", "B-Penalty", "A-HHI", "B-HHI","A-Rank", "B-Rank"]

    v = [penalty_EoM[keys[int(np.where(allowed_integers_EoM == integers[0,i])[0])]],   penalty_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == integers[1,i])[0])]],
        HHI_capacity_EoM[keys[int(np.where(allowed_integers_EoM == integers[0,i])[0])]],   HHI_capacity_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == integers[1,i])[0])]],
        rank_EoM[keys[int(np.where(allowed_integers_EoM == integers[0,i])[0])]],   rank_CM_CfD[keys[int(np.where(allowed_integers_CM_CfD == integers[1,i])[0])]],
        penalty_EoM[keys[int(np.where(allowed_integers_EoM == integers[0,i])[0])]]]

    if i == 4:
        print(v)

    ax.plot(angles, v, label = run_names[i], linewidth=2, zorder = 2, color = colors[colors_index[i]], linestyle = linestyles[i])
    # Add error bars as vertical lines
    # Format the radar chart
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels_not_center, fontsize=10, backgroundcolor="white", zorder=10)
    ax.set_yticks([1, 2, 3])  # Choose which tick values to display
    ax.set_yticklabels(["1.0", "2.0", "3.0"], fontsize=8)
    ax.set_ylim(0, 3)
    for angle, label in zip(angles[:-1], labels_center):
        ax.text(angle, 3.8, label, ha='center', va='center', fontsize=10, zorder=10)
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.2))
    #ax.legend(loc="best")
    ax.set_title('LSTM with MLP in head')

# Adjust spacing
plt.tight_layout()
plt.savefig("training_metrics_spider_all.pdf", format="pdf", bbox_inches="tight")