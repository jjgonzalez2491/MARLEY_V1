import ray
import os
import re
import subprocess
import wandb
from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb
import time
os.environ["RAY_TMPDIR"] = "/dir/tmp/ray"
ray.init(_temp_dir ="/dir/tmp/ray")
import shutil

from ray import air, tune
import numpy as np
from ray.tune.registry import register_env
from CM_CfD import CM_EoM
from ray.rllib.examples.rl_module.action_masking_rlm import TorchActionMaskRLM
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
from action_mask_lstm import CustomPPOCatalog
import pandas as pd

def find_latest_checkpoint(base_path):
    latest_checkpoint = None
    latest_time = 0

    # Traverse the directory tree starting from base_path
    for root, dirs, files in os.walk(base_path):
        for dir_name in dirs:
            # Check if the directory matches the checkpoint pattern
            if re.match(r"^checkpoint_\d+$", dir_name):
                full_path = os.path.join(root, dir_name)
                dir_time = os.path.getmtime(full_path)
                if dir_time > latest_time:
                    latest_checkpoint = full_path
                    latest_time = dir_time

    return latest_checkpoint

# Define the path to the count file
count_file_path = 'run_count_tax_ss_1.txt'

if __name__ == "__main__":

    penalty_type = 3
    
    os.environ['WANDB_DISABLE_SERVICE']= "True"
    wandb.init(project="Test_WB", mode='offline', dir='/dir/wandb_logs')
    # def wandb_callback(result):
        # Log the metrics to W&B
    #    wandb.log(result)# Define the directory where you want to start searching
    
    root_directory = "/dir/ray_results/PPO"

    # Define the prefix of the directories you want to delete
    prefix = "PPO_CM_EoM_ss_battery_1"

    # Walk through the directory tree
    for dirpath, dirnames, filenames in os.walk(root_directory):
        for dirname in dirnames:
            if dirname.startswith(prefix):
                full_path = os.path.join(dirpath, dirname)
                print(f"Deleting directory: {full_path}")
                shutil.rmtree(full_path)
                print(f"Deleted {full_path}")
    
    base_path = "/dir/CM_EoM_ss_battery_1/PPO"
    latest_checkpoint = find_latest_checkpoint(base_path)
    
    if latest_checkpoint:
        print(f"Restoring from checkpoint: {latest_checkpoint}")
        restore_string = latest_checkpoint
        print("Checkpoint restored successfully.")
    else:
        print("No checkpoint found.") 

    """ General system information """

    agent_g = 16

    dir_input_data = '280325 Entry data - Italy 2024 Alice - 16 Agents.xlsx'

    max_t = 144 * 28

    short_t = 24
    
    VoLL = 4000
    
    period_inv = 12 
    
    step_g_bids = 13

    step_g_inv = 5

    opportunity_cost = 0.08

    n_tech = 6

    n_tech_RES = 3

    CfD_activation = True

    CM_activation = True

    random_g = np.random.random([503, agent_g * (n_tech + 3)])

    random_g_cm = np.random.random([503, agent_g * (n_tech + 1)])

    random_g_CfD = np.random.random([503, agent_g * (n_tech_RES)])
    
    v_c_g = pd.read_excel(dir_input_data, sheet_name='Variable_Costs')
    
    v_c_g = np.array(v_c_g.iloc[:,:])
    
    inv_init_g = pd.read_excel(dir_input_data, sheet_name='Agents')
    
    inv_init_g = np.array(inv_init_g.iloc[:,:])
    
    inv_data = pd.read_excel(dir_input_data, sheet_name='Tech_Characteristics')

    inv_max = np.array(inv_data.iloc[0,:])

    life_time = np.array(inv_data.iloc[1,:])

    construction_time = np.array(inv_data.iloc[2,:])

    inv_cost = pd.read_excel(dir_input_data, sheet_name='Inv_Tech')
    
    inv_cost = np.array(inv_cost.iloc[:,:])

    fixed_cost = pd.read_excel(dir_input_data, sheet_name='Fixed_Cost_Tech')
    
    fixed_cost = np.array(fixed_cost.iloc[:,:])

    aging = pd.read_excel(dir_input_data, sheet_name='Aging')

    aging = np.array(aging.iloc[:,:])

    CO2_tax = pd.read_excel(dir_input_data, sheet_name='CO2_tax')

    CO2_tax = np.array(CO2_tax.iloc[:,:])

    CO2_tax_tech = pd.read_excel(dir_input_data, sheet_name='CO2_tax_tech')

    CO2_tax_tech = np.array(CO2_tax_tech.iloc[:,:])

    policy_ids = pd.read_excel(dir_input_data, sheet_name='Policy_IDs')

    policy_ids = np.array(policy_ids.iloc[:,:])

    cm_design = pd.read_excel(dir_input_data, sheet_name='CM_Design')

    cm_price_cap =  np.array(cm_design.iloc[0,0])

    cm_option_strike = np.array(cm_design.iloc[0,1])

    cm_excess_demand = np.array(cm_design.iloc[0,2])
    
    inv_data_battery = pd.read_excel(dir_input_data, sheet_name='Tech_Characteristics_Battery')

    inv_max_battery = np.array(inv_data_battery.iloc[0,:])

    life_time_battery = np.array(inv_data_battery.iloc[1,:])

    construction_time_battery = np.array(inv_data_battery.iloc[2,:])

    inv_cost_battery = pd.read_excel(dir_input_data, sheet_name='Inv_Tech_Battery')
    
    inv_cost_battery = np.array(inv_cost_battery.iloc[:,:])

    fixed_cost_battery = pd.read_excel(dir_input_data, sheet_name='Fixed_Cost_Tech_Battery')
    
    fixed_cost_battery = np.array(fixed_cost_battery.iloc[:,:])

    CfD_price_cap = pd.read_excel(dir_input_data, sheet_name='CfD_Design')

    CfD_price_cap = np.array(CfD_price_cap.iloc[:,:])

    CfD_target = pd.read_excel(dir_input_data, sheet_name='CfD_target')

    CfD_target = np.array(CfD_target.iloc[:,:])

    scenario_failures = pd.read_excel(dir_input_data, sheet_name='Scenarios_Failures')

    scenario_failures = np.array(scenario_failures.iloc[:,:])

    random_number_demand = np.random.randint(0, 20, size=(499))

    random_number_availability = np.random.randint(0, 20, size=(509, n_tech))

    random_number_failures = np.random.randint(0, 20, size=(52100))

    inv_init_battery = pd.read_excel(dir_input_data, sheet_name='Storage_short-term')

    inv_init_battery = np.array(inv_init_battery.iloc[:,0])

    storage_lt_info = pd.read_excel(dir_input_data, sheet_name='Storage_long-term')

    inv_init_storage_lt = np.array(storage_lt_info.iloc[:agent_g,0])

    SoC_max_merchant_storage_lt = np.array(storage_lt_info.iloc[:agent_g,1])

    fixed_cost_storage_lt = pd.read_excel(dir_input_data, sheet_name='Fixed_Cost_Tech_Storage_lt')

    fixed_cost_storage_lt =  np.array(fixed_cost_storage_lt.iloc[:,:])

    investments_enabled = pd.read_excel(dir_input_data, sheet_name='Investments_enabled')

    investments_enabled =  np.array(investments_enabled.iloc[:,:])

    average_failures = pd.read_excel(dir_input_data, sheet_name='Average_failures')

    average_failures =  np.array(average_failures.iloc[:,:])

    time_series = pd.read_excel(dir_input_data, sheet_name='Time_Series')

    time_series =  np.array(time_series.iloc[:,:])

    time_series = time_series[:,2:]

    average_bimester_series = pd.read_excel(dir_input_data, sheet_name='Average_bimester_series')

    average_bimester_series =  np.array(average_bimester_series.iloc[:,:])

    average_bimester_series = average_bimester_series[:,4:]

    average_yearly_series = pd.read_excel(dir_input_data, sheet_name='Average_yearly_series')

    average_yearly_series =  np.array(average_yearly_series.iloc[:,:])

    average_yearly_series = average_yearly_series[:,1:]

    env = CM_EoM( max_t, short_t, VoLL,
			  agent_g, step_g_bids, step_g_inv, n_tech,
			  v_c_g, inv_init_g, inv_max, inv_cost, fixed_cost, construction_time, life_time,
			  aging, CO2_tax_tech, CO2_tax,
			  random_g, random_g_cm, random_g_CfD,
			  cm_price_cap, cm_option_strike, cm_excess_demand,
			  opportunity_cost,
			  n_tech_RES, CfD_price_cap, CfD_target,
			  average_failures,
              scenario_failures,
			  random_number_failures, 
			  inv_init_battery, inv_max_battery, inv_cost_battery, fixed_cost_battery, construction_time_battery, life_time_battery,
			  inv_init_storage_lt, SoC_max_merchant_storage_lt, fixed_cost_storage_lt,
			  investments_enabled,
			  time_series, average_bimester_series, average_yearly_series,
              CfD_activation, CM_activation)

    temp_action_space = env.action_space.sample()

    def env_creator(args):
        
        temp_env = CM_EoM( max_t, short_t, VoLL,
			  agent_g, step_g_bids, step_g_inv, n_tech,
			  v_c_g, inv_init_g, inv_max, inv_cost, fixed_cost, construction_time, life_time,
			  aging, CO2_tax_tech, CO2_tax,
			  random_g, random_g_cm, random_g_CfD,
			  cm_price_cap, cm_option_strike, cm_excess_demand,
			  opportunity_cost,
			  n_tech_RES, CfD_price_cap, CfD_target,
			  average_failures,
              scenario_failures,
			  random_number_failures, 
			  inv_init_battery, inv_max_battery, inv_cost_battery, fixed_cost_battery, construction_time_battery, life_time_battery,
			  inv_init_storage_lt, SoC_max_merchant_storage_lt, fixed_cost_storage_lt,
			  investments_enabled,
			  time_series, average_bimester_series, average_yearly_series,
              CfD_activation, CM_activation)
        
        return temp_env
    
    register_env("CM_EoM_ss_battery_1", env_creator)

    print(policy_ids)

    possible_agents_g = [f"Agent_g_{i}" for i in range (agent_g)]

    possible_policies_g = [f"Policy_g_{i}" for i in range (int(max(policy_ids) + 1))]

    print(possible_policies_g)

    policy_ids_dict = {}

    for i in range (agent_g):
        policy_ids_dict.update({possible_agents_g[i]:possible_policies_g[int(policy_ids[i])]}) 

    print(policy_ids_dict)

    rlm_class = TorchActionMaskRLM

    RL_module_g =  {i: SingleAgentRLModuleSpec(module_class=rlm_class, catalog_class=CustomPPOCatalog) for i in possible_agents_g}

    RL_module_dict = RL_module_g
    
    def policy_mapping_fn(agent_id, episode, worker=None, **kwargs):
        policy_id_agent = policy_ids_dict.get(agent_id)
        return policy_id_agent

    config = (
        PPOConfig()
        .environment("CM_EoM_ss_battery_1",
                     )
        .experimental(
            _enable_new_api_stack=True,
            _disable_preprocessor_api=True,
        )
        .rl_module(
            rl_module_spec=MultiAgentRLModuleSpec(
                module_specs=RL_module_dict
            ),
        )
        .framework("torch")
        .resources(num_gpus_per_learner_worker = 1)
        .rollouts(
            num_rollout_workers = 67, 
            num_envs_per_worker = 8, 
            ) 
        .multi_agent(
            policies=env.get_agent_ids(),
            policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
        )
        .training(
            clip_param = 0.1,
            train_batch_size = 34304,
            sgd_minibatch_size = 34304,
            num_sgd_iter = 10,
            vf_clip_param = float("inf"),
            entropy_coeff = 0.01,
            gamma = 1,
            model={
                "fcnet_hiddens": [512, 512],
                "fcnet_activation": "relu",
            },
            lambda_ = 0.995,
        )
    )   
    
    wandb.config.update(config.to_dict())

    start_time = time.time()

    try:
        tune.run("PPO", name="PPO", 
             stop={"time_total_s": 51480},
             checkpoint_freq=60,  
             checkpoint_at_end=True, 
             config=config.to_dict(), 
             storage_path="/dir/CM_EoM_ss_battery_1",
             restore=restore_string,
             callbacks=[WandbLoggerCallback(project="Test_WB")],
             )

        ray.shutdown()

        wandb.finish()
        
    except Exception as e:
        
        ray.shutdown()
        wandb.finish()
        
        print("SIMULATION ERROR")

    


