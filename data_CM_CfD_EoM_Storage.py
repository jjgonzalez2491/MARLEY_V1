import pandas as pd
import os
import numpy as np
from ray.rllib.policy.policy import Policy
import time
import sys
import numpy_financial as npf
import glob

""" Initializing """

n = 1

from CM_CfD_EoM_Storage import CM_EoM

def find_latest_checkpoint(input_string, CVaR):

    base_path = f"/dir/{input_string}/PPO"

    if not os.path.exists(base_path):
        raise FileNotFoundError(f"The path {base_path} does not exist.")

    # Find all checkpoint files under the base path, regardless of folder name
    checkpoint_paths = glob.glob(os.path.join(base_path, "*/checkpoint_*"))

    if not checkpoint_paths:
        raise FileNotFoundError("No checkpoints found for the given input_string.")

    latest_checkpoint = latest_checkpoint = max(checkpoint_paths, key=os.path.getmtime)

    return latest_checkpoint

scenarios = {
    0: (1, 0, 1, 1, '280325 Entry data - Italy 2024 Alice - 16 Agents.xlsx'),
}

test_idx = n - 1

test_init, cm_excess_demand_tmp, cm_price_cap_tmp, CfD_price_cap_tmp, dir_input_data = scenarios[test_idx]

checkpoint = find_latest_checkpoint(f'CM_EoM_ss_battery_{test_init}', 0)
restored_policies = Policy.from_checkpoint(checkpoint)
print(f"checkpoint_restored{test_init}")

string = f'{test_init}'
 
if __name__ == "__main__":

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

    env = CM_EoM(max_t, short_t, VoLL,
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
			  time_series, average_bimester_series, average_yearly_series)

    """ Variables for saving behavior """
    
    iters_outer_loop = 20
    
    action = {}

    iters_max = int(max_t/short_t)

    obs, info = env.reset()
        
    agent_ids_g = env.possible_agents_g

    weighted_prices = np.zeros([iters_max, iters_outer_loop])

    weighted_prices_reliability = np.zeros([iters_max, iters_outer_loop])

    cm_premium_price = np.zeros([iters_max, iters_outer_loop])

    cm_balance_total = np.zeros([iters_max, iters_outer_loop])

    CfD_premium_price = np.zeros([iters_max, iters_outer_loop])

    CfD_balance_total = np.zeros([iters_max, iters_outer_loop])

    capacity_2030 = np.zeros([n_tech + 1, iters_outer_loop])

    capacity_2030_merchant = np.zeros([n_tech + 1, iters_outer_loop])

    capacity_2030_cm = np.zeros([n_tech + 1, iters_outer_loop])

    capacity_2030_CfD = np.zeros([n_tech + 1, iters_outer_loop])

    capacity_2040 = np.zeros([n_tech + 1, iters_outer_loop])

    capacity_2040_merchant = np.zeros([n_tech + 1, iters_outer_loop])

    capacity_2040_cm = np.zeros([n_tech + 1, iters_outer_loop])

    capacity_2040_CfD = np.zeros([n_tech + 1, iters_outer_loop])

    capacity_2030_agent = np.zeros([agent_g, iters_outer_loop])

    capacity_2040_agent = np.zeros([agent_g, iters_outer_loop])

    capacity_2030_agent_t1 = np.zeros([agent_g, iters_outer_loop])

    capacity_2040_agent_t1 = np.zeros([agent_g, iters_outer_loop])

    capacity_2030_agent_t2 = np.zeros([agent_g, iters_outer_loop])

    capacity_2040_agent_t2 = np.zeros([agent_g, iters_outer_loop])

    capacity_2030_agent_t3 = np.zeros([agent_g, iters_outer_loop])

    capacity_2040_agent_t3 = np.zeros([agent_g, iters_outer_loop])

    capacity_2030_agent_t4 = np.zeros([agent_g, iters_outer_loop])

    capacity_2040_agent_t4 = np.zeros([agent_g, iters_outer_loop])

    capacity_2030_agent_t5 = np.zeros([agent_g, iters_outer_loop])

    capacity_2040_agent_t5 = np.zeros([agent_g, iters_outer_loop])

    capacity_2030_agent_t6 = np.zeros([agent_g, iters_outer_loop])

    capacity_2040_agent_t6 = np.zeros([agent_g, iters_outer_loop])

    capacity_2030_agent_t7 = np.zeros([agent_g, iters_outer_loop])

    capacity_2040_agent_t7 = np.zeros([agent_g, iters_outer_loop])

    CO2_emissions = np.zeros([iters_max, iters_outer_loop])

    prices_hour_2030 = np.zeros([iters_outer_loop, int(short_t*6)])

    prices_hour_2040 = np.zeros([iters_outer_loop, int(short_t*6)])

    capacity_cm = np.zeros([iters_max, n_tech + 1])

    capacity_cm_battery = np.zeros([iters_max])

    capacity_CfD = np.zeros([iters_max, n_tech])

    capacity_merchant_agent = np.zeros([iters_max, agent_g])

    reward_tech_cm = np.zeros([iters_max, n_tech + 1])

    reward_tech_CfD = np.zeros([iters_max, n_tech + 1])

    reward_tech_merchant = np.zeros([iters_max, n_tech + 1])

    reward_tech_existing = np.zeros([iters_max, n_tech + 1])

    price_hour = np.zeros([iters_max,24])

    capacity_merchant = np.zeros([iters_max, n_tech + 1])

    capacity_CfD = np.zeros([iters_max, n_tech])

    capacity_cm_agent = np.zeros([iters_max, agent_g])

    capacity_CfD_agent = np.zeros([iters_max, agent_g])

    total_installed_capacity = np.zeros([iters_max,n_tech + 1 ])

    reward_agent = np.zeros([agent_g, iters_outer_loop])

    SoC_state = np.zeros([iters_max, agent_g])

    SoC_system = np.zeros([iters_max, iters_outer_loop])

    cm_scarcity = np.zeros([iters_max, iters_outer_loop])

    CfD_scarcity = np.zeros([iters_max, iters_outer_loop])

    short_term_scarcity = np.zeros([iters_max, iters_outer_loop])

    reward_agents = np.zeros([iters_max, agent_g])

    """ Looping for market iterations """

    for capacity_iteration in range(iters_outer_loop):

        capacity_2030_agent_temp_tech = np.zeros([agent_g, n_tech + 1])

        capacity_2040_agent_temp_tech = np.zeros([agent_g, n_tech + 1])

        capacity_2030_temp = np.zeros([n_tech + 1])

        capacity_2030_agent_temp = np.zeros([agent_g])

        capacity_2040_temp = np.zeros([n_tech + 1])

        capacity_2040_agent_temp = np.zeros([agent_g])

        action = {}

        obs, info = env.reset()

        i = 0

        price = np.zeros([iters_max])
        CO2_emissions_temp = np.zeros([iters_max])
      
        cm_price = np.zeros([iters_max])
        cm_balance = np.zeros([iters_max])

        CfD_price = np.zeros([iters_max])
        CfD_balance = np.zeros([iters_max])

        # Prices

        hourly_prices = np.zeros([short_t, iters_max])

        i = 0

        delta = 0.001

        t_2030 = 0

        t_2040 = 0  

        states = {} 

        for agent_id in agent_ids_g:
            states[agent_id] = restored_policies.get(agent_id).get_initial_state()

        """ Looping within the market horizon """

        while i < iters_max:
            
            actions = {}
            agents_c = 0

            t = env.time

            for agent_id in agent_ids_g:
        
                policy = restored_policies.get(agent_id)
            
                agent_obs = obs.get(agent_id)
                action,state_out ,_= policy.compute_single_action(agent_obs, states.get(agent_id))
                action_mask = agent_obs.get("action_mask")
                agent_obs = agent_obs.get("observations")

                if max(agent_obs) > 1 or min(agent_obs)  < -1:
                    pass

                actions[agent_id] = action
            
                states[agent_id] = state_out

                agents_c += 1            

            next_obs, reward, done, truncated, _ = env.step(actions)

            cm_price[i] = ((np.sum(env.cm_income) + np.sum(env.cm_income_battery)) /(env.short_t * env.hour_month))/(np.sum(env.cm_inv_options) + np.sum(env.cm_inv_options_battery) + delta) * env.normalization_factor

            cm_balance[i] = env.cm_balance_real/env.percentile_demand

            CfD_income = 0

            CfD_aggregated_capacity = 0

            agents_c = 0

            for agent_id in agent_ids_g:

                reward_agents[i,agents_c] += (np.sum(env.reward_tech_merchant_step[agents_c,:] + env.reward_tech_cm_step[agents_c,:] + env.reward_tech_CfD_step[agents_c,:]) + env.reward_tech_merchant_battery_step[agents_c] + env.reward_tech_cm_battery_step[agents_c])/env.discount_factor

                for tech in range(env.n_tech):

                    CfD_income += env.CfD_price_agents[agents_c,tech] * env.capacity_CfD[agents_c,tech] * env.availability_tech_average_yearly[env.year_int, tech]

                    CfD_aggregated_capacity +=  env.capacity_CfD[agents_c,tech]  * env.availability_tech_average_yearly[env.year_int, tech]

                agents_c += 1

            CfD_price[i] = CfD_income/(CfD_aggregated_capacity + delta)

            CfD_balance[i] = env.CfD_balance_real/env.percentile_demand

            price[i] = env.price_net

            capacity_cm[i,:n_tech] += np.sum(env.capacity_cm, axis=0)/iters_outer_loop

            capacity_cm[i,n_tech] += np.sum(env.capacity_cm_battery, axis=0)/iters_outer_loop

            total_installed_capacity[i, :n_tech] += np.sum(env.inv_g, axis=0)/iters_outer_loop

            total_installed_capacity[i, n_tech] += np.sum(env.inv_g_battery)/iters_outer_loop

            capacity_cm_battery[i] += np.sum(env.capacity_cm_battery)/iters_outer_loop

            capacity_merchant[i,:n_tech] += np.sum(env.capacity_merchant, axis=0)/iters_outer_loop 

            capacity_merchant[i,n_tech] += np.sum(env.capacity_merchant_battery, axis=0)/iters_outer_loop 

            capacity_cm_agent[i,:] += np.sum(env.capacity_cm, axis=1)/iters_outer_loop

            capacity_merchant_agent[i,:] += np.sum(env.capacity_merchant, axis=1)/iters_outer_loop

            capacity_CfD[i,:] += np.sum(env.capacity_CfD, axis=0)/iters_outer_loop

            capacity_CfD_agent[i,:] += np.sum(env.capacity_CfD, axis=1)/iters_outer_loop

            for t in range(24):

                price_hour[i,t] += env.hourly_prices[t]/iters_outer_loop
                if env.hourly_prices[t] > 1000:
                    short_term_scarcity[env.year_int,capacity_iteration] += 1
                
            if env.year == 1:
                pass

            elif env.year_int == 12:
                capacity_2030_temp[:env.n_tech] += np.sum(env.inv_g, axis = 0)/6
                capacity_2030_agent_temp += np.sum(env.inv_g, axis = 1)/6 
                capacity_2030_agent_temp += (env.capacity_merchant_battery + env.capacity_cm_battery)/6
                capacity_2030_temp[env.n_tech] += np.sum(env.inv_g_battery)/6
                capacity_2030_agent_temp_tech[:,:(env.n_tech)] += env.inv_g/6
                capacity_2030_agent_temp_tech[:,env.n_tech] += (env.capacity_merchant_battery + env.capacity_cm_battery)/6

                # Capacity merchant

                capacity_2030_merchant[:env.n_tech, capacity_iteration] += np.sum(env.capacity_merchant, axis = 0)/6
                capacity_2030_merchant[env.n_tech, capacity_iteration] += np.sum(env.capacity_merchant_battery)/6

                # Capacity cm

                capacity_2030_cm[:env.n_tech, capacity_iteration] += np.sum(env.capacity_cm, axis = 0)/6
                capacity_2030_cm[env.n_tech, capacity_iteration] += np.sum(env.capacity_cm_battery)/6

                # Capacity CfD

                capacity_2030_CfD[:env.n_tech, capacity_iteration] += np.sum(env.capacity_CfD, axis = 0)/6
                capacity_2030_CfD[env.n_tech, capacity_iteration] += 0

                for t in range(24):
                    prices_hour_2030[capacity_iteration,t_2030] = env.hourly_prices[t]
                    t_2030 += 1

            elif env.year_int == 22:
                capacity_2040_temp[:env.n_tech] += np.sum(env.inv_g, axis = 0)/6
                capacity_2040_agent_temp += np.sum(env.inv_g, axis = 1)/6 
                capacity_2040_agent_temp += (env.capacity_merchant_battery + env.capacity_cm_battery)/6
                capacity_2040_temp[env.n_tech] += np.sum(env.inv_g_battery)/6
                capacity_2040_agent_temp_tech[:,:(env.n_tech)] += env.inv_g/6
                capacity_2040_agent_temp_tech[:,env.n_tech] += (env.capacity_merchant_battery + env.capacity_cm_battery)/6
                
                for t in range(24):
                    prices_hour_2040[capacity_iteration,t_2040] = env.hourly_prices[t]
                    t_2040 += 1

                # Capacity merchant

                capacity_2040_merchant[:env.n_tech, capacity_iteration] += np.sum(env.capacity_merchant, axis = 0)/6
                capacity_2040_merchant[env.n_tech, capacity_iteration] += np.sum(env.capacity_merchant_battery)/6

                # Capacity cm

                capacity_2040_cm[:env.n_tech, capacity_iteration] += np.sum(env.capacity_cm, axis = 0)/6
                capacity_2040_cm[env.n_tech, capacity_iteration] += np.sum(env.capacity_cm_battery)/6

                # Capacity CfD

                capacity_2040_CfD[:env.n_tech, capacity_iteration] += np.sum(env.capacity_CfD, axis = 0)/6
                capacity_2040_CfD[env.n_tech, capacity_iteration] += 0

            for tech in range(env.n_tech):
                reward_tech_merchant[i, tech] += (np.sum(env.reward_tech_merchant_step[:,tech])/iters_outer_loop)/env.discount_factor
                reward_tech_cm[i,tech] += (np.sum(env.reward_tech_cm_step[:,tech])/iters_outer_loop)/env.discount_factor
                reward_tech_existing[i, tech] += (np.sum(env.reward_tech_existing_step[:,tech])/iters_outer_loop)/env.discount_factor
                reward_tech_CfD[i,tech] += (np.sum(env.reward_tech_CfD_step[:,tech])/iters_outer_loop)/env.discount_factor
                
            reward_tech_merchant[i, tech + 1] += (np.sum(env.reward_tech_merchant_battery_step)/iters_outer_loop)/env.discount_factor
            reward_tech_cm[i,tech + 1] += (np.sum(env.reward_tech_cm_battery_step)/iters_outer_loop)/env.discount_factor

            CO2_emissions_temp[i] = env.CO2_emissions_step

            SoC_state[i,:] += (env.SoC_merchant_storage_lt)/(env.SoC_max_merchant_storage_lt + delta)/iters_outer_loop

            SoC_system[i, capacity_iteration] = np.sum(env.SoC_merchant_storage_lt)/(np.sum(env.SoC_max_merchant_storage_lt) + delta)

            obs = next_obs

            cm_scarcity[i,capacity_iteration] = env.cm_scarcity

            CfD_scarcity[i,capacity_iteration] = env.CfD_scarcity

            i += 1
            
        ## Data from iterations 

        weighted_prices[:, capacity_iteration] = price

        cm_premium_price[:, capacity_iteration] = cm_price

        cm_balance_total[:, capacity_iteration] = cm_balance

        CfD_premium_price[:, capacity_iteration] = CfD_price

        CfD_balance_total[:, capacity_iteration] = CfD_balance

        capacity_2030[:, capacity_iteration] = capacity_2030_temp

        capacity_2040[:, capacity_iteration] = capacity_2040_temp

        capacity_2030_agent[:, capacity_iteration] = capacity_2030_agent_temp

        capacity_2040_agent[:, capacity_iteration] = capacity_2040_agent_temp

        CO2_emissions[:, capacity_iteration] = CO2_emissions_temp

        reward_agent[:,capacity_iteration] = np.sum(env.reward_tech_existing, axis = 1) + np.sum(env.reward_tech_merchant, axis = 1) + np.sum(env.reward_tech_cm, axis = 1) + np.sum(env.reward_tech_CfD, axis = 1) + env.reward_tech_merchant_battery + env.reward_tech_cm_battery + env.reward_tech_existing_storage_lt

        capacity_2030_agent_t1[:, capacity_iteration] = capacity_2030_agent_temp_tech[:, 0]

        capacity_2040_agent_t1[:, capacity_iteration] = capacity_2040_agent_temp_tech[:, 0]

        capacity_2030_agent_t2[:, capacity_iteration] = capacity_2030_agent_temp_tech[:, 1]

        capacity_2040_agent_t2[:, capacity_iteration] = capacity_2040_agent_temp_tech[:, 1]

        capacity_2030_agent_t3[:, capacity_iteration] = capacity_2030_agent_temp_tech[:, 2]

        capacity_2040_agent_t3[:, capacity_iteration] = capacity_2040_agent_temp_tech[:, 2]

        capacity_2030_agent_t4[:, capacity_iteration] = capacity_2030_agent_temp_tech[:, 3]

        capacity_2040_agent_t4[:, capacity_iteration] = capacity_2040_agent_temp_tech[:, 3]

        capacity_2030_agent_t5[:, capacity_iteration] = capacity_2030_agent_temp_tech[:, 4]

        capacity_2040_agent_t5[:, capacity_iteration] = capacity_2040_agent_temp_tech[:, 4]

        capacity_2030_agent_t6[:, capacity_iteration] = capacity_2030_agent_temp_tech[:, 5]

        capacity_2040_agent_t6[:, capacity_iteration] = capacity_2040_agent_temp_tech[:, 5]

        capacity_2030_agent_t7[:, capacity_iteration] = capacity_2030_agent_temp_tech[:, 6]

        capacity_2040_agent_t7[:, capacity_iteration] = capacity_2040_agent_temp_tech[:, 6]

    IRR_tech_merchant = np.zeros([env.n_tech + 1])

    IRR_tech_cm = np.zeros([env.n_tech + 1])

    IRR_tech_CfD = np.zeros([env.n_tech + 1])
    
    for tech in range(env.n_tech + 1):

        IRR_tech_merchant[tech] = (1 + npf.irr(reward_tech_merchant[:,tech])) ** 6 - 1

        IRR_tech_merchant[tech] = IRR_tech_merchant[tech] if np.isfinite(IRR_tech_merchant[tech]) else 100

        IRR_tech_merchant[tech] = IRR_tech_merchant[tech] if np.sum(capacity_merchant[:,tech]) > 0 else 200

        IRR_tech_cm[tech] = (1 + npf.irr(reward_tech_cm[:,tech])) ** 6 - 1

        IRR_tech_cm[tech] = IRR_tech_cm[tech] if np.isfinite(IRR_tech_cm[tech]) else 100

        IRR_tech_cm[tech] = IRR_tech_cm[tech] if np.sum(capacity_cm[:,tech]) > 0 else 200

        IRR_tech_CfD[tech] = (1 + npf.irr(reward_tech_CfD[:,tech])) ** 6 - 1

        IRR_tech_CfD[tech] = IRR_tech_CfD[tech] if np.isfinite(IRR_tech_CfD[tech]) else 100

        if tech < (env.n_tech):

            capacity_CfD_temp = np.sum(capacity_CfD[:,tech])

        else:

            capacity_CfD_temp =  0

        IRR_tech_CfD[tech] = IRR_tech_CfD[tech] if capacity_CfD_temp > 0 else 200

    IRR_agents = np.zeros([agent_g])

    agents_c = 0

    for agent_id in agent_ids_g:

        IRR_agents[agents_c] = (1 + npf.irr(reward_agents[:,agents_c])) ** 6 - 1 

        agents_c += 1

    """ Saving information """

    # Check if the directory exists, if not, create it
    if not os.path.exists(f'results/{string}'):
        os.makedirs(f'results/{string}')

    df = pd.DataFrame((SoC_system))
    df.to_csv(f'results/{string}/SoC_system.csv')

    df = pd.DataFrame((SoC_state))
    df.to_csv(f'results/{string}/SoC_agent.csv')

    df = pd.DataFrame((weighted_prices))
    df.to_csv(f'results/{string}/prices.csv')

    df = pd.DataFrame((prices_hour_2030))
    df.to_csv(f'results/{string}/prices_hour_2030.csv')

    df = pd.DataFrame((prices_hour_2040))
    df.to_csv(f'results/{string}/prices_hour_2040.csv')

    df = pd.DataFrame((price_hour))
    df.to_csv(f'results/{string}/prices_hour.csv')

    df = pd.DataFrame((cm_premium_price))
    df.to_csv(f'results/{string}/cm_premium_price.csv')

    df = pd.DataFrame((cm_balance_total))
    df.to_csv(f'results/{string}/cm_balance_total.csv')

    df = pd.DataFrame((CfD_premium_price))
    df.to_csv(f'results/{string}/CfD_premium_price.csv')

    df = pd.DataFrame(CfD_balance_total)
    df.to_csv(f'results/{string}/CfD_balance_total.csv')

    df = pd.DataFrame((capacity_2030))
    df.to_csv(f'results/{string}/capacity_2030.csv')

    df = pd.DataFrame((capacity_2030_merchant))
    df.to_csv(f'results/{string}/capacity_2030_merchant.csv')

    df = pd.DataFrame((capacity_2030_cm))
    df.to_csv(f'results/{string}/capacity_2030_cm.csv')

    df = pd.DataFrame((capacity_2030_CfD))
    df.to_csv(f'results/{string}/capacity_2030_CfD.csv')

    df = pd.DataFrame((capacity_2040))
    df.to_csv(f'results/{string}/capacity_2040.csv')

    df = pd.DataFrame((capacity_2040_merchant))
    df.to_csv(f'results/{string}/capacity_2040_merchant.csv')

    df = pd.DataFrame((capacity_2040_cm))
    df.to_csv(f'results/{string}/capacity_2040_cm.csv')

    df = pd.DataFrame((capacity_2040_CfD))
    df.to_csv(f'results/{string}/capacity_2040_CfD.csv')

    df = pd.DataFrame((capacity_2030_agent))
    df.to_csv(f'results/{string}/capacity_2030_agent.csv')

    df = pd.DataFrame((capacity_2040_agent))
    df.to_csv(f'results/{string}/capacity_2040_agent.csv')

    df = pd.DataFrame((capacity_merchant))
    df.to_csv(f'results/{string}/capacity_merchant.csv')

    df = pd.DataFrame((capacity_cm))
    df.to_csv(f'results/{string}/capacity_cm.csv')

    df = pd.DataFrame((capacity_cm_battery))
    df.to_csv(f'results/{string}/capacity_cm_battery.csv')

    df = pd.DataFrame((capacity_CfD))
    df.to_csv(f'results/{string}/capacity_CfD.csv')

    df = pd.DataFrame((CO2_emissions))
    df.to_csv(f'results/{string}/CO2_emissions.csv')
    
    df = pd.DataFrame((total_installed_capacity))
    df.to_csv(f'results/{string}/EoM_E_TIC.csv')

    df = pd.DataFrame((reward_agent))
    df.to_csv(f'results/{string}/Reward_agent.csv')

    df = pd.DataFrame((cm_scarcity))
    df.to_csv(f'results/{string}/cm_scarcity.csv')

    df = pd.DataFrame((CfD_scarcity))
    df.to_csv(f'results/{string}/CfD_scarcity.csv')

    df = pd.DataFrame((capacity_2030_agent_t1))
    df.to_csv(f'results/{string}/capacity_2030_agent_t1.csv')

    df = pd.DataFrame((capacity_2040_agent_t1))
    df.to_csv(f'results/{string}/capacity_2040_agent_t1.csv')

    df = pd.DataFrame((capacity_2030_agent_t2))
    df.to_csv(f'results/{string}/capacity_2030_agent_t2.csv')

    df = pd.DataFrame((capacity_2040_agent_t2))
    df.to_csv(f'results/{string}/capacity_2040_agent_t2.csv')

    df = pd.DataFrame((capacity_2030_agent_t3))
    df.to_csv(f'results/{string}/capacity_2030_agent_t3.csv')

    df = pd.DataFrame((capacity_2040_agent_t3))
    df.to_csv(f'results/{string}/capacity_2040_agent_t3.csv')

    df = pd.DataFrame((capacity_2030_agent_t4))
    df.to_csv(f'results/{string}/capacity_2030_agent_t4.csv')

    df = pd.DataFrame((capacity_2040_agent_t4))
    df.to_csv(f'results/{string}/capacity_2040_agent_t4.csv')

    df = pd.DataFrame((capacity_2030_agent_t5))
    df.to_csv(f'results/{string}/capacity_2030_agent_t5.csv')

    df = pd.DataFrame((capacity_2040_agent_t5))
    df.to_csv(f'results/{string}/capacity_2040_agent_t5.csv')

    df = pd.DataFrame((capacity_2030_agent_t6))
    df.to_csv(f'results/{string}/capacity_2030_agent_t6.csv')

    df = pd.DataFrame((capacity_2040_agent_t6))
    df.to_csv(f'results/{string}/capacity_2040_agent_t6.csv')

    df = pd.DataFrame((capacity_2030_agent_t7))
    df.to_csv(f'results/{string}/capacity_2030_agent_t7.csv')

    df = pd.DataFrame((capacity_2040_agent_t7))
    df.to_csv(f'results/{string}/capacity_2040_agent_t7.csv')

    df = pd.DataFrame((short_term_scarcity))
    df.to_csv(f'results/{string}/short_term_scarcity.csv')

    df = pd.DataFrame((IRR_tech_cm))
    df.to_csv(f'results/{string}/IRR_cm.csv')

    df = pd.DataFrame((IRR_tech_CfD))
    df.to_csv(f'results/{string}/IRR_CfD.csv')

    df = pd.DataFrame((IRR_tech_merchant))
    df.to_csv(f'results/{string}/IRR_merchant.csv')

    df = pd.DataFrame((IRR_agents))
    df.to_csv(f'results/{string}/IRR_agents.csv')    
    
    
    









