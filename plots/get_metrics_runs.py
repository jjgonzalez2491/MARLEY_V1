import os
import numpy as np
import pandas as pd

# Path to the parent directory containing folders

# data_path = '/dir/291224_results'

data_path = '/dir/301224_results'

# Initialize list to hold matrices
Reward_agents = np.zeros([27,16,100])

reward_agent_1 = np.zeros([27])
percentile_reward_agent_1 = np.zeros([27,2])

reward_agent_8 = np.zeros([27])
percentile_reward_agent_8 = np.zeros([27,2])

total_reward = np.zeros([27])

total_reward_percentile = np.zeros([27,2])

total_reward_incumbent = np.zeros([27])

total_reward_entrants = np.zeros([27])

capacity_2030_agents = np.zeros([27,16,100])

capacity_2040_agents = np.zeros([27,16,100])

total_number_broke = np.zeros([27])

penalty_agents = np.zeros([27,16,100])

total_penalty = np.zeros([27])

total_penalty_percentile = np.zeros([27,2])

total_number_without_penalty = np.zeros([27])

HHI_capacity_2030 = np.zeros([27])

HHI_capacity_2040 = np.zeros([27])

percentile_HHI_capacity_2040 = np.zeros([27,2])

HHI_rewards_2040 = np.zeros([27])

HHI_rewards_2040_incumbent = np.zeros([27])

HHI_rewards_2040_entrants = np.zeros([27])

total_number_without_penalty = np.zeros([27])

total_emissions = np.zeros([27])

total_emissions_percentile = np.zeros([27,2])

print(f"Shape of 3D matrix: {Reward_agents.shape}")

start = 1
end = 27

base_string = "_3_cm_storage_mask_lstm_"

folder_names = []

# run_names = np.array([46, 47, 48, 52, 53, 56, 57, 60, 61, 64, 79, 80, 81, 82, 83, 89, 91, 93, 95, 97, 99, 101, 103, 105, 107, 109, 111])

run_names = np.array([66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 84, 85, 86, 87, 88, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130])

for i in range(start, end + 1):
    folder = f"{i}{base_string}{run_names[i-1]}"
    print(folder)

# Read each CSV and append to matrices list
for i in range(start, end + 1):

    folder = f"{i}{base_string}{run_names[i-1]}"

    rewards = pd.read_csv(os.path.join(data_path, folder, 'Reward_Penalty_agent.csv')) # Read as numpy array
    rewards = np.array(rewards.iloc[:,1:])
    Reward_agents[i-1,:,:] = rewards
    total_reward[i-1] = np.sum(rewards)
    vector_reward = np.sum(rewards, axis = 0)

    total_reward_percentile[i-1,0] = total_reward[i-1]/100 - np.percentile(vector_reward,10)
    total_reward_percentile[i-1,1] = np.percentile(vector_reward,90) - total_reward[i-1]/100       
    
    total_reward_incumbent[i-1] = np.sum(rewards[:8])
    total_reward_entrants[i-1] = np.sum(rewards[-8:])

    reward_agent_1[i-1] = np.sum(rewards[1,:])
    reward_agent_1_vector = rewards[1,:]
    percentile_reward_agent_1[i-1,0] = reward_agent_1[i-1]/100 - np.percentile(reward_agent_1_vector,10)
    percentile_reward_agent_1[i-1,1] = np.percentile(reward_agent_1_vector,90) - reward_agent_1[i-1]/100 

    reward_agent_8[i-1] = np.sum(rewards[8,:])
    reward_agent_8_vector = rewards[8,:]
    percentile_reward_agent_8[i-1,0] = reward_agent_8[i-1]/100 - np.percentile(reward_agent_8_vector,10)
    percentile_reward_agent_8[i-1,1] = np.percentile(reward_agent_8_vector,90) -  reward_agent_8[i-1]/100 


    capacity_2030 = pd.read_csv(os.path.join(data_path, folder, 'capacity_2030_agent.csv')) # Read as numpy array
    capacity_2030 = np.array(capacity_2030.iloc[:,1:])
    capacity_2030_agents[i-1,:,:] = capacity_2030

    capacity_2040 = pd.read_csv(os.path.join(data_path, folder, 'capacity_2040_agent.csv')) # Read as numpy array
    capacity_2040 = np.array(capacity_2040.iloc[:,1:])
    capacity_2040_agents[i-1,:,:] = capacity_2040

    HHI_2030_tmp_capacity = np.zeros([100])
    HHI_2040_tmp_capacity = np.zeros([100])

    HHI_2040_tmp_reward = np.zeros([100])

    HHI_2040_tmp_reward_incumbent = np.zeros([100])
    HHI_2040_tmp_reward_entrants = np.zeros([100])

    for j in range(100):

        total_capacity_2030 = np.sum(capacity_2030[:,j])
        total_capacity_2040 = np.sum(capacity_2040[:,j])

        total_reward_2040 = np.sum(rewards[:,j][rewards[:,j] > 0])

        for k in range(16):
            HHI_2030_tmp_capacity[j] += ((capacity_2030[k,j]/total_capacity_2030)*100)**2
            HHI_2040_tmp_capacity[j] += ((capacity_2040[k,j]/total_capacity_2040)*100)**2
            HHI_2040_tmp_reward[j] += ((np.maximum(rewards[k,j]/total_reward_2040,0))*100)**2

            if k < 8:
                HHI_2040_tmp_reward_incumbent[j] += ((np.maximum(rewards[k,j]/total_reward_2040,0))*100)**2

            else:
                HHI_2040_tmp_reward_entrants[j] += ((np.maximum(rewards[k,j]/total_reward_2040,0))*100)**2

    HHI_capacity_2030[i-1] = np.average(HHI_2030_tmp_capacity)
    HHI_capacity_2040[i-1] = np.average(HHI_2040_tmp_capacity)

    percentile_HHI_capacity_2040[i-1,0] =  HHI_capacity_2040[i-1] - np.percentile(HHI_2040_tmp_capacity,10)
    percentile_HHI_capacity_2040[i-1,1] =  np.percentile(HHI_2040_tmp_capacity,90) - HHI_capacity_2040[i-1]

    HHI_rewards_2040[i-1] = np.average(HHI_2040_tmp_reward)

    HHI_rewards_2040_entrants[i-1] = np.average(HHI_2040_tmp_reward_entrants)
    HHI_rewards_2040_incumbent[i-1] = np.average(HHI_2040_tmp_reward_incumbent)

    penalty = pd.read_csv(os.path.join(data_path, folder, 'Penalty_term_agent.csv')) # Read as numpy array
    penalty = np.array(penalty.iloc[:,1:])
    penalty_agents[i-1,:,:] = penalty
    total_penalty[i-1] = np.sum(penalty)

    vector_penalty = np.sum(penalty, axis = 0)

    total_penalty_percentile[i-1,0] = total_penalty[i-1]/100 - np.percentile(vector_penalty,10)
    total_penalty_percentile[i-1,1] = np.percentile(vector_penalty,90) - total_penalty[i-1]/100       

    emissions = pd.read_csv(os.path.join(data_path, folder, 'CO2_emissions.csv')) # Read as numpy array
    emissions = np.array(emissions.iloc[:,1:])
    total_emissions[i-1] = np.sum(emissions)/(10**9)

    vector_emissions = np.sum(emissions, axis = 0)/(10**9)

    total_emissions_percentile[i-1,0] = total_emissions[i-1]/100 - np.percentile(vector_emissions,10)
    total_emissions_percentile[i-1,1] = np.percentile(vector_emissions,90) - total_emissions[i-1]/100 

    total_number_broke[i-1] = np.sum(rewards < 0)/(100*16)
    total_number_without_penalty[i-1] = np.sum(penalty > 0)/(100*16)

print(f"Shape of 3D matrix: {Reward_agents.shape}")

summary_matrix = np.vstack([total_reward, total_reward_incumbent, total_reward_entrants, total_penalty, HHI_rewards_2040, 
                            HHI_rewards_2040_incumbent, HHI_rewards_2040_entrants, HHI_capacity_2030, HHI_capacity_2040,
                            total_number_broke, total_number_without_penalty, total_emissions])


string = '030325_analysis'

df = pd.DataFrame((summary_matrix/100))
df.to_csv(f'{string}/summary_matrix.csv')

df = pd.DataFrame((total_reward/100))
df.to_csv(f'{string}/total_reward.csv')

df = pd.DataFrame((total_reward_percentile))
df.to_csv(f'{string}/total_reward_percentile.csv')

df = pd.DataFrame((total_penalty_percentile))
df.to_csv(f'{string}/total_penalty_percentile.csv')


df = pd.DataFrame((reward_agent_1/100))
df.to_csv(f'{string}/reward_agent_1.csv')

df = pd.DataFrame((reward_agent_8/100))
df.to_csv(f'{string}/reward_agent_8.csv')

df = pd.DataFrame((percentile_reward_agent_1))
df.to_csv(f'{string}/percentile_reward_agent_1.csv')

df = pd.DataFrame((percentile_reward_agent_8))
df.to_csv(f'{string}/percentile_reward_agent_8.csv')


df = pd.DataFrame((total_emissions_percentile))
df.to_csv(f'{string}/total_emissions_percentile.csv')

df = pd.DataFrame((total_reward_incumbent/100))
df.to_csv(f'{string}/total_reward_incumbent.csv')

df = pd.DataFrame((total_reward_entrants/100))
df.to_csv(f'{string}/total_reward_entrants.csv')

df = pd.DataFrame(total_penalty/100)
df.to_csv(f'{string}/total_penalty.csv')

df = pd.DataFrame(HHI_rewards_2040)
df.to_csv(f'{string}/HHI_rewards_2040.csv')

df = pd.DataFrame(HHI_rewards_2040_incumbent)
df.to_csv(f'{string}/HHI_rewards_2040_incumbents.csv')

df = pd.DataFrame(HHI_rewards_2040_entrants)
df.to_csv(f'{string}/HHI_rewards_2040_entrants.csv')

df = pd.DataFrame((HHI_capacity_2030))
df.to_csv(f'{string}/HHI_capacity_2030.csv')

df = pd.DataFrame((HHI_capacity_2040))
df.to_csv(f'{string}/HHI_capacity_2040.csv')

df = pd.DataFrame((percentile_HHI_capacity_2040))
df.to_csv(f'{string}/percentile_HHI_capacity_2040.csv')

df = pd.DataFrame((total_number_broke/100))
df.to_csv(f'{string}/total_broke.csv')

df = pd.DataFrame((total_number_without_penalty/100))
df.to_csv(f'{string}/total_number_without_penalty.csv')

df = pd.DataFrame((total_emissions/100))
df.to_csv(f'{string}/total_emissions.csv')






