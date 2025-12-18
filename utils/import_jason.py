## Import data from JSON files

import json
import numpy as np
import pandas as pd
import os
from pathlib import Path
 
def find_values(key, dictionary):
    values = []

    def _find_values(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == key:
                    values.append(v)
                elif isinstance(v, (dict, list)):
                    _find_values(v)
        elif isinstance(obj, list):
            for item in obj:
                _find_values(item)

    _find_values(dictionary)
    return values


string = '071124_Runs_8_2'

agent_g = 16

def find_root_folders(root_folder):
    
    # List to store paths of json files
    json_files = []

    # Walk through directory structure
    for dirpath, _, filenames in os.walk(root_folder):
        # Check if the directory path starts with the specified structure
        if dirpath.startswith(f"{root_folder}/PPO_CM_EoM_ss_battery_46"):
            for filename in filenames:
                if filename == "result.json":
                    file_path = os.path.join(dirpath, filename)
                    json_files.append(file_path)

    # Sort the files by creation time
    json_files.sort(key=lambda x: os.path.getctime(x))

    return json_files

root_folder = '/dir/CM_EoM_ss_battery_46/PPO'

# Read the file

file_names = find_root_folders(root_folder)

agent_ids_g = [f"Agent_g_{i}" for i in range (agent_g)]

number_agents = len(agent_ids_g)

j = 0

for file_name in file_names:

    print(file_name)

    with open(file_name, 'r') as file:
        content = file.read()

    # Split the content by new lines or other delimiters if necessary
    json_objects = content.split('\n')  # assuming each JSON object is on a new line

    # Parse each JSON object
    data_list = []
    for obj in json_objects:
        if obj.strip():  # Skip empty lines
            data_list.append(json.loads(obj))

    # Now, data_list contains all JSON objects

    key_to_find = 'counters'
    counter = find_values(key_to_find, data_list)

    key_to_find = 'num_env_steps_sampled'
    num_env_steps_sampled = find_values(key_to_find, counter)
        
    # Specify the key you want to search for
    key_to_find = 'sampler_results'
    sampler_results = find_values(key_to_find, data_list)

    key_to_find_e_min = 'episode_reward_min'
    episode_reward_min = find_values(key_to_find_e_min, sampler_results)

    key_to_find_e_mean = 'episode_reward_mean'
    episode_reward_mean = find_values(key_to_find_e_mean, sampler_results)

    key_to_find_e_max = 'episode_reward_max'
    episode_reward_max = find_values(key_to_find_e_max, sampler_results)

    key_info = 'info'
    info = find_values(key_info, data_list)

    key_learner = 'learner'
    learner = find_values(key_learner, info)

    if j == 0:

        num_env_steps_sampled_aggregate = num_env_steps_sampled[3:]
        episode_reward_min_aggregated = episode_reward_min[3:]
        episode_reward_mean_aggregated = episode_reward_mean[3:]
        episode_reward_max_aggregated = episode_reward_max[3:]

        num_env_steps_sampled_aggregate_n = num_env_steps_sampled

    else:

        num_env_steps_sampled_aggregate = np.concatenate((num_env_steps_sampled_aggregate,num_env_steps_sampled[3:]))
        num_env_steps_sampled_aggregate_n = np.concatenate((num_env_steps_sampled_aggregate_n,num_env_steps_sampled))

        episode_reward_min_aggregated = np.concatenate((episode_reward_min_aggregated,  episode_reward_min[3:]))
        episode_reward_mean_aggregated = np.concatenate((episode_reward_mean_aggregated,  episode_reward_mean[3:]))
        episode_reward_max_aggregated = np.concatenate((episode_reward_max_aggregated,  episode_reward_max[3:]))

    key_to_find_min = 'policy_reward_min'
    policy_reward_min = find_values(key_to_find_min, sampler_results)

    key_to_find_mean = 'policy_reward_mean'
    policy_reward_mean = find_values(key_to_find_mean, sampler_results)

    key_to_find_max = 'policy_reward_max'
    policy_reward_max = find_values(key_to_find_max, sampler_results)

    i = 0

    for agent_id in agent_ids_g:

        key_to_find = agent_id

        policy_reward_min_agent = find_values(key_to_find, policy_reward_min)

        policy_reward_mean_agent = find_values(key_to_find, policy_reward_mean)

        policy_reward_max_agent = find_values(key_to_find, policy_reward_max)

        agent_var = find_values(key_to_find, learner)

        key_to_find_var = 'vf_explained_var'

        var = find_values(key_to_find_var, agent_var)

        if i == 0:

            policy_reward_min_a = policy_reward_min_agent
            policy_reward_mean_a = policy_reward_mean_agent
            policy_reward_max_a = policy_reward_max_agent

            var_a = var

        else:

            policy_reward_min_a = np.vstack((policy_reward_min_a, policy_reward_min_agent))
            policy_reward_mean_a = np.vstack((policy_reward_mean_a, policy_reward_mean_agent))
            policy_reward_max_a = np.vstack((policy_reward_max_a, policy_reward_max_agent))

            var_a = np.vstack((var_a, var))

        i += 1
    
    if j == 0:

        policy_reward_min_aggregate = np.transpose(policy_reward_min_a)
        policy_reward_mean_aggregate = np.transpose(policy_reward_mean_a)
        policy_reward_max_aggregate = np.transpose(policy_reward_max_a)

        var_aggregate = np.transpose(var_a)
    
    else:

        policy_reward_min_aggregate = np.vstack((policy_reward_min_aggregate,  np.transpose(policy_reward_min_a)))
        policy_reward_mean_aggregate = np.vstack((policy_reward_mean_aggregate,  np.transpose(policy_reward_mean_a)))
        policy_reward_max_aggregate = np.vstack((policy_reward_max_aggregate,  np.transpose(policy_reward_max_a)))       

        var_aggregate = np.vstack((var_aggregate,  np.transpose(var_a)))       

    j += 1


if not os.path.exists(string):
    os.makedirs(string)

df = pd.DataFrame(policy_reward_min_aggregate)
df.to_csv(f'{string}/agent_min.csv')

df = pd.DataFrame(np.transpose(np.vstack((num_env_steps_sampled_aggregate, np.transpose(policy_reward_mean_aggregate)))))
df.to_csv(f'{string}/agent_mean.csv')

df = pd.DataFrame(policy_reward_max_aggregate)
df.to_csv(f'{string}/agent_max.csv')

df = pd.DataFrame(episode_reward_min_aggregated)
df.to_csv(f'{string}/episode_reward_min.csv')

df = pd.DataFrame(np.transpose(np.vstack((num_env_steps_sampled_aggregate, episode_reward_mean_aggregated))))
df.to_csv(f'{string}/episode_reward_mean.csv')

df = pd.DataFrame(episode_reward_max_aggregated)
df.to_csv(f'{string}/episode_reward_max.csv')

df = pd.DataFrame(num_env_steps_sampled_aggregate_n)
df.to_csv(f'{string}/env_steps_n.csv')

df = pd.DataFrame(num_env_steps_sampled_aggregate)
df.to_csv(f'{string}/env_steps.csv')

df = pd.DataFrame(var_aggregate)
df.to_csv(f'{string}/var_agents.csv')
