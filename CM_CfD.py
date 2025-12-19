import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium.spaces import Dict, Box, MultiDiscrete
from collections import OrderedDict
import copy

"""
Long-term market environment

"""

class CM_EoM(MultiAgentEnv):
	def __init__(self, max_t, short_t, VoLL,
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
              CfD_activation, CM_activation):
		
		""" Market activation """

		# Activation of CfD markets
		self.CfD_activation = CfD_activation

		# Activation of CM markets
		self.CM_activation = CM_activation
		
		""" Termination and truncation """
		
		self.terminateds = set()
		self.truncateds = set()	

		""" Time Constants and initialization """

		# Init time

		self.max_t_o = max_t
		self.t_init = 0
		self.max_t = max_t

		self.planning_horizon = 4

		# Short term time constants

		self.short_t = short_t

		self.yearly_resolution = 6

		# Simulation year and month

		self.year = 0
		self.year_int = 0
		self.month = 0
		self.hour_year = 0
		self.hour_month = 62.4

		# Maximum year

		self.max_year = self.max_t/(self.short_t * self.yearly_resolution)

		# Initial time and maximum horizon

		self.time = 0

		self.time_random = 0 
		
		self.max_t = self.max_t_o

		# Random numbers for failure matrix 

		self.scenario_failures = scenario_failures

		self.random_number_failures = random_number_failures

		self.time_random_failures = 0

		""" Demand information """

		# Adjusting demand to the starting year

		self.demand = time_series[:,8]
		self.max_demand = np.max(self.demand)

		self.demand_average_bimester = average_bimester_series[:,8]
		self.demand_average_year = average_yearly_series[:,8]

		self.percentile_demand = np.percentile(self.demand,90) 

		""" CO2 tax and emission information information """

		self.CO2_tax_original = CO2_tax
		self.CO2_tax_tech_original = CO2_tax_tech

		self.CO2_tax = CO2_tax
		self.CO2_tax_tech = CO2_tax_tech
		self.CO2_emissions_step = 0

		""" Agent initialization """

		# Number of agents

		self.agents_g = agent_g
		self.agents_n = self.agents_g 

		# Agent ids

		self.possible_agents_g = [f"Agent_g_{i}" for i in range (self.agents_g)]
		self.agents = self.possible_agents_g
		self.possible_agents = self.agents
		self._agent_ids = set(self.possible_agents)

		self.under_construction = {}

		for a in self.possible_agents_g:
			# Notation: tech, capacity, time, merchant/Capacity market, capacity price
			self.under_construction.update({a:np.zeros([1,6])})

		""" Investments on generation assets """
		
		# Number of technologies

		self.n_tech = n_tech

		# Aging 
  
		self.aging = aging

		# Monthly installments corresponding to investment costs

		self.inv_cost = inv_cost

		# Fixed costs

		self.fixed_cost = fixed_cost

		# Construction time for technologies and construction vector

		self.construction_time = construction_time

		# Project Lifetime

		self.life_time = life_time		
		
		# Setting investment to initial investment 

		self.inv_init_g = inv_init_g
		self.inv_g = copy.deepcopy(self.inv_init_g)
		self.inv_max = inv_max
		self.inv_new = np.zeros([self.agents_g, self.n_tech])
		self.inv_age = np.zeros([self.agents_g, self.n_tech])
		self.inv_age_merchant = np.zeros([self.agents_g, self.n_tech])
		self.capacity_merchant = np.zeros([self.agents_g, self.n_tech]) 
		self.inv_g_aging = copy.deepcopy(self.inv_init_g)

		self.inv_age_cm = np.zeros([self.agents_g, self.n_tech])
		self.inv_age_CfD = np.zeros([self.agents_g, self.n_tech])
		self.capacity_cm = np.zeros([self.agents_g, self.n_tech])
		self.capacity_merchant_uc = np.zeros([self.agents_g, self.n_tech])
		self.capacity_cm_uc = np.zeros([self.agents_g, self.n_tech])
		
		self.inv_cost_accum_merchant = np.zeros([self.agents_g, self.n_tech])
		self.inv_cost_accum = np.zeros([self.agents_g, self.n_tech])
		self.inv_cost_accum_cm = np.zeros([self.agents_g, self.n_tech])
		self.inv_cost_accum_CfD = np.zeros([self.agents_g, self.n_tech])		
		
		# GENCOs' parameters (Variable costs, initial investment, max investment, and investment costs)

		self.v_c_g = v_c_g
		self.availability_tech = time_series[:,:8]
		self.availability_tech_average_bimester = average_bimester_series[:,:8]
		self.availability_tech_average_yearly = average_yearly_series[:,:8]

		## Reward per technology

		self.reward_tech_merchant = np.zeros([self.agents_g, self.n_tech])
		self.reward_tech_cm = np.zeros([self.agents_g, self.n_tech])
		self.reward_tech_CfD = np.zeros([self.agents_g, self.n_tech])
		self.reward_tech_existing = np.zeros([self.agents_g, self.n_tech])

		## Resetting accumulated profit

		self.accumulated_profit_cm = np.zeros([self.n_tech])
		self.accumulated_profit_merchant = np.zeros([self.n_tech])
		self.accumulated_profit_CfD= np.zeros([self.n_tech])

		## Capacity market construction

		self.cm_fc_under_construction = np.zeros([self.n_tech])

		""" Investments on storage assets """		
		
		# Battery investment variables 

		self.capacity_merchant_battery = np.zeros([self.agents_g]) 
		self.capacity_merchant_battery_uc = np.zeros([self.agents_g])  

		self.inv_init_storage_lt = inv_init_storage_lt

		self.capacity_existing_storage_lt = self.inv_init_storage_lt
		self.capacity_existing_storage_lt_uc = np.zeros([self.agents_g]) 
		self.SoC_max_merchant_storage_lt = SoC_max_merchant_storage_lt

		self.SoC_merchant_storage_lt = self.SoC_max_merchant_storage_lt * 0.5

		self.inv_age_merchant_battery = np.zeros([self.agents_g])
		self.inv_cost_accum_merchant_battery = np.zeros([self.agents_g])

		self.capacity_cm_battery = np.zeros([self.agents_g]) 
		self.capacity_cm_battery_uc = np.zeros([self.agents_g]) 
		self.inv_age_cm_battery = np.zeros([self.agents_g])
		self.inv_cost_accum_cm_battery = np.zeros([self.agents_g])
		
		# Battery costs and lifetime

		self.inv_init_battery = inv_init_battery

		self.inv_max_battery = inv_max_battery

		self.inv_cost_battery = inv_cost_battery

		self.fixed_cost_battery = fixed_cost_battery
		self.fixed_cost_storage_lt = fixed_cost_storage_lt

		self.construction_time_battery = construction_time_battery 

		self.life_time_battery = life_time_battery

		# Charging strategies

		self.strategy_charge_3_hours = np.zeros([self.short_t])
		self.strategy_discharge_3_hours = np.zeros([self.short_t])

		self.strategy_charge_4_hours = np.zeros([self.short_t])
		self.strategy_discharge_4_hours = np.zeros([self.short_t])

		self.strategy_charge_8_hours = np.zeros([self.short_t])
		self.strategy_discharge_8_hours = np.zeros([self.short_t])

		self.strategy_charge_cm = np.zeros([self.short_t])
		self.strategy_discharge_cm = np.zeros([self.short_t])

		self.inv_init_battery = inv_init_battery
		self.inv_g_battery =  copy.deepcopy(self.inv_init_battery)

		# Reward and accumulated profit - Battery

		self.reward_tech_merchant_battery = np.zeros([self.agents_g])
		self.reward_tech_merchant_battery_step = np.zeros([self.agents_g])

		self.reward_tech_existing_storage_lt = np.zeros([self.agents_g])
		self.reward_tech_existing_storage_lt_step = np.zeros([self.agents_g])

		self.reward_tech_cm_battery = np.zeros([self.agents_g])
		self.reward_tech_cm_battery_step = np.zeros([self.agents_g])

		self.accumulated_profit_merchant_battery = 0
		self.accumulated_profit_cm_battery = 0

		## Capacity market construction

		self.cm_fc_under_construction_battery = 0

		""" Availability and failure information """
		
		## Failure values

		self.average_failures = average_failures
		
		""" Agents particular information """

		# Investments enabled

		self.investments_enabled = investments_enabled		
		
		# Opportunity cost

		self.opportunity_cost = opportunity_cost
		self.opportunity_cost_merchant = opportunity_cost
		self.opportunity_cost_cm = opportunity_cost
		self.opportunity_cost_CfD = opportunity_cost

		""" Market initialization and constants """

		# Market information

		self.VoLL = VoLL
		self.price_avg = self.VoLL * np.random.random()/50
		self.cm_price_cap = cm_price_cap
		self.cm_price = 0
		self.cm_option_strike = cm_option_strike
		self.cm_auction_indicator = False
		self.cm_excess_demand = cm_excess_demand
		self.CfD_price = 0
		self.CfD_price_agents = np.zeros([self.agents_g, self.n_tech])
		self.CfD_auction_indicator = False

		# Capacity valuation, capacity credtis for market and firm capactiy

		self.cm_tech_cc = np.ones([self.n_tech]) * 0.9

		# Income from Capacity Market

		self.cm_income = np.zeros([self.agents_g, self.n_tech])

		# Capacity with options coming from the capacity market

		self.cm_inv_options = np.zeros([self.agents_g, self.n_tech])

		# CfD auctions

		self.CfD_price_cap = CfD_price_cap
		self.CfD_target_original = CfD_target
		self.CfD_target = CfD_target

		# Scarcity signals

		self.cm_scarcity = 0
		self.CfD_scarcity = 0

		""" Initialization of actions spaces """
				
		# Discretization steps for MultiDiscrete action spaces (step_g_p should be 11 for strategic bidding)

		self.step_g_bids = step_g_bids
		self.step_g_inv = step_g_inv

		self.step_SoC_control = 7
		
		# Number of investments bid plus investment actions
		
		self.act_inv = self.n_tech + 1

		# Actions for capacity market auctions

		self.act_bids_cm = self.n_tech + 1
		self.act_inv_cm = self.n_tech + 1

		# Actions for CfD auctions

		self.n_tech_RES = n_tech_RES

		self.act_bids_CfD = self.n_tech_RES
		self.act_inv_CfD = self.n_tech_RES

		# Actions for SoC lt

		self.act_SoC = 1

		# Number of total bids

		self.n_act_g = self.act_inv + self.act_bids_cm + self.act_inv_cm + self.act_bids_CfD + self.act_inv_CfD + self.act_SoC

		# Action spaces 
		
		self._action_space_in_preferred_format = True

		action_matrix = np.concatenate((np.ones([self.act_inv])*self.step_g_inv, 
								  np.ones([self.act_bids_cm]) * self.step_g_bids, np.ones([self.act_inv_cm])*self.step_g_inv,
								  np.ones([self.act_bids_CfD]) * self.step_g_bids, np.ones([self.act_inv_CfD])*self.step_g_inv,
								   np.ones([self.act_SoC]) * self.step_SoC_control,
								  ),axis=0)

		action_spaces_g = {i: MultiDiscrete(action_matrix) for i in self.possible_agents_g}
		self.action_space = Dict(action_spaces_g)

		""" Initialization of Observation spaces """

		# Observation spaces (General and Specific observations)

		self.n_observations_general = self.short_t

		self.n_observation_specific = 0

		self.n_observation_resources = 10

		self.n_observation_specific_2 = 13

		self.n_observation_specific_2_battery = 9

		self.n_observation_time = 4		

		self.n_observation_cm_CfD = 6

		self.n_observation_reward_tech = 4

		self.n_observation_reward_tech_CfD = 2

		self.n_observation_reward_tech_battery = 4

		self.n_obs_g = self.n_observations_general + (self.n_tech * self.n_observation_specific) + self.n_observation_resources + (self.n_tech * self.n_observation_specific_2) + self.n_observation_specific_2_battery + self.n_observation_time + self.n_observation_cm_CfD + (self.n_observation_reward_tech * self.n_tech) + (self.n_observation_reward_tech_CfD * self.n_tech_RES) + self.n_observation_reward_tech_battery
		
		self._obs_space_in_preferred_format = True

		observation_spaces_g = {
            i: Dict({"observations": Box(low = np.ones(self.n_obs_g) * -1, high = np.ones(self.n_obs_g), dtype=np.float16),
					 "action_mask": Box(0.0, 1.0, shape=(int(self.step_g_inv * self.act_inv + self.step_g_bids * self.act_bids_cm + self.step_g_inv * self.act_inv_cm
											  + self.step_g_bids * self.act_bids_CfD + self.step_g_inv * self.act_inv_CfD
											  + self.step_SoC_control * self.act_SoC),), 
										 dtype=np.float16),})
            for i in self.possible_agents_g
        }

		self.observation_space = Dict(observation_spaces_g)

		""" Constants and random variable matrices """

		# Constant to avoid divison by zero

		self.delta_small = 0.001		
		
		# Random values for bids 

		self.random_g_p = random_g		
		self.random_g_cm = random_g_cm		
		self.random_g_CfD = random_g_CfD

		# Reward normalization

		self.normalization_factor = self.VoLL * self.VoLL * self.max_t_o

		""" Init method for compatibility with RLLIB """

		super().__init__()
		
	def reset(self, *, seed=None, options=None):

		""" CO2 tax and emission information information """

		self.CO2_tax = self.CO2_tax_original 

		self.CO2_tax_tech = self.CO2_tax_tech_original 

		self.CO2_emissions_step = 0

		""" Time Constants and initialization """

		# Init time

		self.t_init = 0

		# Initial time and maximum horizon

		self.time = 0
		
		self.max_t = self.max_t_o

		self.count_last = np.zeros([self.n_tech])

		self.count_last_battery = 0

		self.year_int = 0

		# Simulation year and month

		self.year = 0

		self.month = 0

		self.hour_year = 0

		""" Agents particular information """

		# Discount Factor

		self.discount_factor = 1/((1 + self.opportunity_cost) ** self.year)
		self.discount_factor_merchant =  1/((1 + (self.opportunity_cost_merchant)) ** self.year)
		self.discount_factor_cm = self.discount_factor
		self.discount_factor_CfD = self.discount_factor

		""" Investments on generation assets """

		# Reseting investments to initial conditions

		self.inv_g = copy.deepcopy(self.inv_init_g)
		self.inv_g_aging = copy.deepcopy(self.inv_init_g)
		
		self.under_construction = {}

		for a in self.possible_agents_g:
			
			self.under_construction.update({a:np.zeros([1,6])})

		# Investment quantities

		self.inv_new = np.zeros([self.agents_g, self.n_tech])
		self.inv_age = np.zeros([self.agents_g, self.n_tech])
		self.inv_age_cm = np.zeros([self.agents_g, self.n_tech])
		self.inv_age_CfD = np.zeros([self.agents_g, self.n_tech])
		self.inv_age_merchant = np.zeros([self.agents_g, self.n_tech])

		self.capacity_merchant = np.zeros([self.agents_g, self.n_tech])
		self.capacity_cm = np.zeros([self.agents_g, self.n_tech])
		self.capacity_CfD = np.zeros([self.agents_g, self.n_tech])
		self.capacity_merchant_uc = np.zeros([self.agents_g, self.n_tech])
		self.capacity_cm_uc = np.zeros([self.agents_g, self.n_tech])
		self.capacity_CfD_uc = np.zeros([self.agents_g, self.n_tech])

		self.inv_cost_accum = np.zeros([self.agents_g, self.n_tech])
		self.inv_cost_accum_merchant = np.zeros([self.agents_g, self.n_tech])
		self.inv_cost_accum_cm = np.zeros([self.agents_g, self.n_tech])
		self.inv_cost_accum_CfD = np.zeros([self.agents_g, self.n_tech])

		self.cm_fc_under_construction = np.zeros([self.n_tech])
		self.cm_fc_under_construction_battery_merchant = 0
		self.cm_fc_under_construction_battery_cm = 0
		self.cm_income = np.zeros([self.agents_g, self.n_tech])
		self.cm_inv_options = np.zeros([self.agents_g, self.n_tech])

		self.CfD_fc_under_construction = np.zeros([self.n_tech])
		self.CfD_income = np.zeros([self.agents_g, self.n_tech])
		self.CfD_price_pond = np.zeros([self.agents_g, self.n_tech])

		## Reward per technology

		self.reward_tech_merchant = np.zeros([self.agents_g, self.n_tech])
		self.reward_tech_cm = np.zeros([self.agents_g, self.n_tech])
		self.reward_tech_existing = np.zeros([self.agents_g, self.n_tech])
		self.reward_tech_CfD = np.zeros([self.agents_g, self.n_tech])

		## Resetting accumulated profit

		self.accumulated_profit_cm = np.zeros([self.n_tech])
		self.accumulated_profit_merchant = np.zeros([self.n_tech])
		self.accumulated_profit_CfD = np.zeros([self.n_tech])

		""" Investments on storage assets """

		self.cm_income_battery = np.zeros([self.agents_g])
		self.cm_inv_options_battery = np.zeros([self.agents_g])

		# Investment quantities

		self.capacity_merchant_battery = np.zeros([self.agents_g]) 

		self.capacity_merchant_battery_uc = np.zeros([self.agents_g]) 

		self.inv_age_merchant_battery = np.zeros([self.agents_g])

		self.inv_cost_accum_merchant_battery = np.zeros([self.agents_g])

		self.inv_g_battery =  copy.deepcopy(self.inv_init_battery)

		self.capacity_cm_battery = np.zeros([self.agents_g]) 
		self.capacity_cm_battery_uc = np.zeros([self.agents_g]) 

		self.inv_age_cm_battery = np.zeros([self.agents_g])
		self.inv_cost_accum_cm_battery = np.zeros([self.agents_g])

		self.capacity_existing_storage_lt = self.inv_init_storage_lt
		self.capacity_existing_storage_lt_uc = np.zeros([self.agents_g]) 

		self.SoC_merchant_storage_lt = self.SoC_max_merchant_storage_lt * 0.5

		# Rewards 

		self.reward_tech_merchant_battery = np.zeros([self.agents_g])
		self.reward_tech_merchant_battery_step = np.zeros([self.agents_g])

		self.reward_tech_existing_storage_lt = np.zeros([self.agents_g])
		self.reward_tech_existing_storage_lt_step = np.zeros([self.agents_g])

		self.reward_tech_cm_battery = np.zeros([self.agents_g])
		self.reward_tech_cm_battery_step = np.zeros([self.agents_g])
		
		self.accumulated_profit_merchant_battery = 0
		self.accumulated_profit_cm_battery = 0
 
		""" Availability and demand per scenario """

		self.scenario = 0

		index_init = (self.year_int) * self.short_t * self.yearly_resolution * 5 + self.month * self.short_t * 5 + self.scenario * self.short_t

		index_final = (self.year_int) * self.short_t * self.yearly_resolution * 5 + self.month* self.short_t * 5 + (self.scenario + 1) * self.short_t

		self.availability_tech_step = self.availability_tech[int(index_init): int(index_final),:]
 
		self.demand_step = self.demand[int(index_init): int(index_final)]
						
		# Demand projection

		average_demand_projection_short = self.demand_average_bimester[int(self.year_int * self.yearly_resolution + self.month)]
		average_demand_projection_long = self.demand_average_year[self.year_int + self.planning_horizon]
		
		# Resource availability (short)

		average_solar_availability_short = self.availability_tech_average_bimester[int(self.year_int * self.yearly_resolution + self.month), 0]

		average_wind_availability_short = self.availability_tech_average_bimester[int(self.year_int * self.yearly_resolution + self.month), 1]

		average_hydro_availability_short = self.availability_tech_average_bimester[int(self.year_int * self.yearly_resolution + self.month), self.n_tech]

		average_hydro_ror_availability_short = self.availability_tech_average_bimester[int(self.year_int * self.yearly_resolution + self.month), self.n_tech + 1]

		average_solar_availability_long = self.availability_tech_average_yearly[self.year_int, 0]

		average_wind_availability_long = self.availability_tech_average_yearly[self.year_int, 1]

		average_hydro_availability_long = self.availability_tech_average_yearly[self.year_int, self.n_tech]

		average_hydro_ror_availability_long = self.availability_tech_average_yearly[self.year_int, self.n_tech + 1]

		""" Market information """

		self.CfD_target = self.CfD_target_original

		# Scarcity signals

		self.cm_scarcity = 0
		self.CfD_scarcity = 0		
		
		## Feeling up relevant observations after reset

		self.cm_price = 0
		self.cm_auction_indicator = False
		self.cm_balance_real = 0

		self.cm_balance, self.cm_auction_indicator, self.cm_tech_cc, self.hourly_balance, self.cm_balance_real = self.cm_balance_estimation()
		
		self.CfD_price = 0
		self.CfD_auction_indicator = False
		self.CfD_balance_real = 0 

		self.CfD_balance, self.CfD_auction_indicator, self.CfD_balance_real  = self.CfD_balance_estimation()

		self.CfD_price_agents = np.zeros([self.agents_g, self.n_tech])

		""" Observations for restart """

		observations = self.observation_space.sample()

		obs_temp = np.zeros([self.agents_g, self.n_obs_g])

		i = 0

		for a in self.possible_agents_g:

			## Prices 
			
			for t in range(self.short_t):
				
				obs_temp[i,t] = np.clip(((50 * 2)/(self.VoLL) - 1), -1, 1)

			## Resource availability

			# Solar availability 
			obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific) + 0] = np.clip((average_solar_availability_short * 2) - 1, -1, 1)
					
			# Wind availability
			obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific) + 1] = np.clip((average_wind_availability_short * 2) - 1, -1, 1)
					
			# Hydro availability
			obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific) + 2] = np.clip((average_hydro_availability_short * 2) - 1, -1, 1)

			# Solar availability 
			obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific) + 3] = np.clip((average_solar_availability_long * 2) - 1, -1, 1)
					
			# Wind availability
			obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific) + 4] = np.clip((average_wind_availability_long * 2) - 1, -1, 1)
					
			# Hydro availability
			obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific) + 5] = np.clip((average_hydro_availability_long * 2) - 1, -1, 1)

			# Demand projection (short)
			obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific) + 6] = np.clip((average_demand_projection_short/self.max_demand * 2) - 1, -1, 1)

			# Demand projection (short)
			obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific) + 7] = np.clip((average_demand_projection_long/self.max_demand * 2) - 1, -1, 1)

			# Hydro ROR with respect to Demand projection (short)
			obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific) + 8] = np.clip((average_demand_projection_short - average_hydro_ror_availability_short)/(average_demand_projection_short + average_hydro_ror_availability_short), -1, 1)

			# Hydro ROR with respect to Demand projection (long)
			obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific) + 9] = np.clip((average_demand_projection_long - average_hydro_ror_availability_long)/(average_demand_projection_long + average_hydro_ror_availability_long), -1, 1)

			## Installed capacity (individual and total)

			for j in range(self.n_tech):

				ind_installed_capacity = self.inv_g[i,j]

				ind_total_installed_capacity = np.sum(self.inv_g[i,:])

				total_installed_capacity = np.sum(self.inv_g[:,j])

				ind_installed_capacity_existing = self.inv_g_aging[i,j]

				total_installed_capacity_existing = np.sum(self.inv_g_aging[:,j])

				ind_installed_capacity_merchant = self.capacity_merchant[i,j]

				total_installed_capacity_merchant = np.sum(self.capacity_merchant[:,j])

				ind_installed_capacity_cm = self.capacity_cm[i,j]

				total_installed_capacity_cm = np.sum(self.capacity_cm[:,j])

				ind_installed_capacity_CfD = self.capacity_CfD[i,j]

				total_installed_capacity_CfD = np.sum(self.capacity_CfD[:,j])

				## Demand projections with respect to install technologies

				# Individual	
						
				obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific)  + self.n_observation_resources + (self.n_observation_specific_2 * j) + 0] = np.clip((average_demand_projection_long/self.agents_g - ind_installed_capacity)/(average_demand_projection_long/self.agents_g + ind_installed_capacity), -1, 1)

				# Individual	
						
				obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific)  + self.n_observation_resources + (self.n_observation_specific_2 * j) + 1] = np.clip((average_demand_projection_long - total_installed_capacity)/(average_demand_projection_long + total_installed_capacity), -1, 1)
				
				# Individual	
						
				obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific)  + self.n_observation_resources + (self.n_observation_specific_2 * j) + 2] = np.clip((ind_installed_capacity)/(ind_total_installed_capacity + self.delta_small) * 2 - 1, -1, 1)

				# Total

				obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific)  + self.n_observation_resources + (self.n_observation_specific_2 * j) + 3] = np.clip((ind_installed_capacity)/(total_installed_capacity + self.delta_small) * 2 - 1, -1, 1)

				# ind - Existing

				obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific)  + self.n_observation_resources + (self.n_observation_specific_2 * j) + 4] = np.clip(ind_installed_capacity_existing / (ind_installed_capacity + self.delta_small) * 2 - 1, -1, 1)

				# total - existing

				obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific)  + self.n_observation_resources + (self.n_observation_specific_2 * j) + 5] = np.clip((total_installed_capacity_existing)/(total_installed_capacity + self.delta_small) * 2 - 1, -1, 1)

				# ind - Merchant

				obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific)  + self.n_observation_resources + (self.n_observation_specific_2 * j) + 6] = np.clip(ind_installed_capacity_merchant / (ind_installed_capacity + self.delta_small) * 2 - 1, -1, 1)

				# total - Merchant

				obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific)  + self.n_observation_resources + (self.n_observation_specific_2 * j) + 7] = np.clip((total_installed_capacity_merchant)/(total_installed_capacity + self.delta_small) * 2 - 1, -1, 1)

				# ind - CM

				obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific)  + self.n_observation_resources + (self.n_observation_specific_2 * j) + 8] = np.clip(ind_installed_capacity_cm / (ind_installed_capacity + self.delta_small) * 2 - 1, -1, 1)

				# total - CM

				obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific)  + self.n_observation_resources + (self.n_observation_specific_2 * j) + 9] = np.clip((total_installed_capacity_cm)/(total_installed_capacity + self.delta_small) * 2 - 1, -1, 1)

				# ind - CfD

				obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific)  + self.n_observation_resources + (self.n_observation_specific_2 * j) + 10] = np.clip(ind_installed_capacity_CfD / (ind_installed_capacity + self.delta_small) * 2 - 1, -1, 1)

				# total - CfD

				obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific)  + self.n_observation_resources + (self.n_observation_specific_2 * j) + 11] = np.clip((total_installed_capacity_CfD)/(total_installed_capacity + self.delta_small) * 2 - 1, -1, 1)

				# Capacity credits - CM

				obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific)  + self.n_observation_resources + (self.n_observation_specific_2 * j) + 12] = np.clip((self.cm_tech_cc[0,j]) * 2 - 1, -1, 1)

			# Battery observations

			ind_installed_capacity_battery = self.inv_g_battery[i]

			total_installed_capacity_battery = np.sum(self.inv_g_battery)

			# Individual	
						
			obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific) + self.n_observation_resources + (self.n_observation_specific_2 * self.n_tech) + 0] = np.clip((average_demand_projection_long/self.agents_g - ind_installed_capacity_battery)/(average_demand_projection_long/self.agents_g + ind_installed_capacity_battery), -1, 1)

			# Total	
						
			obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific) + self.n_observation_resources + (self.n_observation_specific_2 * self.n_tech) + 1]= np.clip((average_demand_projection_long - total_installed_capacity_battery)/(average_demand_projection_long + total_installed_capacity_battery), -1, 1)

			## Time observations

			# Month

			obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific) + self.n_observation_resources + (self.n_observation_specific_2 * self.n_tech) + self.n_observation_specific_2_battery + 0] = np.clip((self.month/self.yearly_resolution) * 2 - 1, -1, 1)
   
			# Year
   
			obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific) + self.n_observation_resources + (self.n_observation_specific_2 * self.n_tech) + self.n_observation_specific_2_battery + 1] = np.clip((self.year/self.max_year) * 2 - 1, -1, 1)
   
			# Time
   
			obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific) + self.n_observation_resources + (self.n_observation_specific_2 * self.n_tech) + self.n_observation_specific_2_battery + 2] = np.clip(self.hour_year/(self.max_year * self.yearly_resolution) * 2 - 1, -1, 1)

			# CO2 tax
   
			obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific) + self.n_observation_resources + (self.n_observation_specific_2 * self.n_tech) + self.n_observation_specific_2_battery + 3] = np.clip((self.CO2_tax[self.year_int]/300) * 2 - 1, -1, 1)

			## Capacity Market and CfD observations

			# CM Balance 

			obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific) + self.n_observation_resources + (self.n_observation_specific_2 * self.n_tech) + self.n_observation_specific_2_battery + self.n_observation_time + 0] = np.clip(self.cm_balance_real/self.percentile_demand,-1,1)

			# CM price

			obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific) + self.n_observation_resources + (self.n_observation_specific_2 * self.n_tech) + self.n_observation_specific_2_battery + self.n_observation_time + 1] = np.clip((((np.sum(self.cm_income) * self.normalization_factor /(self.short_t * self.hour_month))/(np.sum(self.cm_inv_options) + self.delta_small))/self.cm_price_cap) * 2 - 1, -1, 1)

			# CM scarcity

			obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific) + self.n_observation_resources + (self.n_observation_specific_2 * self.n_tech) + self.n_observation_specific_2_battery + self.n_observation_time + 2] = np.clip((self.cm_scarcity * 2) - 1, -1, 1)

			# CfD Balance 

			obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific) + self.n_observation_resources + (self.n_observation_specific_2 * self.n_tech) + self.n_observation_specific_2_battery + self.n_observation_time + 3] = np.clip(self.CfD_balance_real/self.percentile_demand,-1,1)

			# CfD Price 

			obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific) + self.n_observation_resources + (self.n_observation_specific_2 * self.n_tech) + self.n_observation_specific_2_battery + self.n_observation_time + 4] = np.clip(((np.sum(self.CfD_price_pond)/(np.sum(self.capacity_CfD) + self.delta_small))/self.CfD_price_cap) * 2 - 1, -1, 1)

			# CfD scarcity

			obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific) + self.n_observation_resources + (self.n_observation_specific_2 * self.n_tech) + self.n_observation_specific_2_battery + self.n_observation_time + 5] = np.clip((self.CfD_scarcity * 2) - 1, -1, 1)

			## Adding observations to dictionary

			observations_temp = {"observations": np.float16(obs_temp[i,:]),
							"action_mask":self.get_action_masks_g(self.investments_enabled[i,:], self.capacity_existing_storage_lt[i])} 
			
			observations.update({a:observations_temp}) 

			i += 1

		""" Resetting terminateds, truncateds, and infos """

		self.terminateds = set()
		self.truncateds = set() 
		
		# Empty info vector

		infos = {a: {} for a in self.agents}
		
		return observations, infos  # reward, done, info can't be included
		
	def step(self, action_dict):

		""" Variables for correct stepping """
		
		# Info, terminated, truncated

		info = {a: {} for a in self.agents}
		terminated = {a: False for a in self.agents}
		truncated = {a: False for a in self.agents}

		# Collecting actions
		actions_g = []
		for i in self.possible_agents_g:
			actions_g.append(action_dict[i])

		action_temp = np.zeros([self.agents_n,  self.n_act_g])
		
		# Rewards
			
		reward_temp = np.zeros(self.agents_g)

		rewards_g = {a: 0 for a in self.possible_agents_g}

		# Observations

		obs_temp = np.zeros([self.agents_g, self.n_obs_g])

		observations_temp = OrderedDict()
		observations_temp = {"observations": np.float32(obs_temp[0,:]),
						"action_mask":self.get_action_masks_g(self.investments_enabled[0,:], self.capacity_existing_storage_lt[0])} 
		
		# Initial observation vector

		observations_g = {a: observations_temp for a in self.possible_agents_g}
		
		""" Resetting variables  for step """

		# year integer

		self.year_int = int(np.floor(self.year))

		# Discount Factor

		self.discount_factor = 1/((1 + self.opportunity_cost) ** self.year)
		self.discount_factor_merchant =  1/((1 + (self.opportunity_cost_merchant)) ** self.year)
		self.discount_factor_cm = self.discount_factor
		self.discount_factor_CfD = self.discount_factor

		# Calculate aggregated failures

		self.aggregated_failures = self.calculate_aggregated_failures()

		# Emissions

		self.CO2_emissions_step = 0

		self.hourly_prices = np.zeros(self.short_t)

		""" Scenario for demand and resource availability """

		# Random scenario selection

		self.scenario = np.random.randint(0, 4)

		# Availability and demand per scenario

		index_init = (self.year_int) * self.short_t * self.yearly_resolution * 5 + self.month * self.short_t * 5 + self.scenario * self.short_t
		index_final = (self.year_int) * self.short_t * self.yearly_resolution * 5 + self.month* self.short_t * 5 + (self.scenario + 1) * self.short_t

		self.availability_tech_step = self.availability_tech[int(index_init): int(index_final),:]
 
		self.demand_step = self.demand[int(index_init): int(index_final)]

		""" Reward step resetting """

		# Generation assets

		self.reward_tech_merchant_step = np.zeros([self.agents_g, self.n_tech])
		self.reward_tech_cm_step = np.zeros([self.agents_g, self.n_tech])
		self.reward_tech_CfD_step = np.zeros([self.agents_g, self.n_tech])
		self.reward_tech_existing_step = np.zeros([self.agents_g, self.n_tech])

		# Storage assets

		self.reward_tech_merchant_battery_step = np.zeros([self.agents_g])
		self.reward_tech_cm_battery_step = np.zeros([self.agents_g])
		self.reward_tech_existing_storage_lt_step = np.zeros([self.agents_g])
	
		""" Capacity market auctions """

		p_g_cm = np.zeros([self.agents_g, self.act_bids_cm])
		q_g_cm = np.zeros([self.agents_g, self.act_bids_cm])

		## Capacity market auction (in case auction indicator was activated in the last period)

		if self.cm_auction_indicator and self.month == 3 and self.CM_activation:

			# Looping through agents and technologies to get quantity and price bids to the markets

			i = 0

			for a in self.possible_agents_g:

				action_temp[i,:] = actions_g[i]  

				for j in range(self.act_bids_cm):
					
					# Selecting maximum investment

					if j < self.n_tech: 

						max_q_cm = self.inv_max[j] 
					else: 

						max_q_cm = self.inv_max_battery

					# Price bids for capacity market
					p_g_cm[i,j] = (action_temp[i, self.act_inv + j] - 1)/(self.step_g_bids - 2) * self.cm_price_cap
						
					# Quantity bids for capacity market
					q_g_cm[i, j] = ((action_temp[i, self.act_inv + self.act_bids_cm + j] - 1)/(self.step_g_inv - 2)) * (max_q_cm * self.cm_tech_cc[0,j])
					
				i += 1

			p_g_cm = np.where(p_g_cm < 0, 0, p_g_cm)

			q_g_cm = np.where(q_g_cm < 0, 0, q_g_cm)

			p_cm_demand = np.ones(2) * self.cm_price_cap

			q_cm_demand = np.ones(2) * max(self.cm_balance, 0)/2

			q_g_cm_flatt = q_g_cm.flatten()

			p_g_cm_flatt = p_g_cm.flatten()
	
			# Running double-sided auction (Capacity Market)
						
			cm_price_flatt, q_accepted_u_cm, q_accepted_g_cm_flatt, self.cm_scarcity = self.double_side_auction_pay_as_bid(p_cm_demand,
																				  q_cm_demand, p_g_cm_flatt, q_g_cm_flatt, self.random_g_cm, self.cm_price_cap)
			
			q_accepted_g_cm = q_accepted_g_cm_flatt.reshape((self.agents_g, self.act_bids_cm))

			self.cm_price = cm_price_flatt.reshape((self.agents_g, self.act_bids_cm))

			i = 0

			for a in self.possible_agents_g:

				inv_under_construction = self.under_construction.get(a)

				for j in range(self.act_bids_cm):

					# Vector for new investment (if action was taken to invest). All winning projects are built

					cc_inv_temp = q_accepted_g_cm[i, j]
					cc_inv_total = q_g_cm[i, j]/(self.cm_tech_cc[0,j] + self.delta_small)

					if j < self.n_tech: 

						construction_time = self.construction_time[j]

						inv_cost = self.inv_cost[self.year_int, j]

					else: 

						construction_time = self.construction_time_battery

						inv_cost = self.inv_cost_battery[self.year_int]
							
					if cc_inv_temp > 0 and self.max_year - self.year >  construction_time and self.cm_tech_cc[0,j] > 0.02:

						inv_vector = np.zeros([1,6])
						inv_vector[0,0] = j
						inv_vector[0,1] = cc_inv_total  
						inv_vector[0,2] = construction_time
						inv_vector[0,3] = 1
						inv_vector[0,4] = self.cm_price[i,j]

						inv_vector[0,5] = cc_inv_total * inv_cost / self.normalization_factor

						inv_under_construction = np.vstack([inv_under_construction, inv_vector])

						if j < self.n_tech: 
						
							# Firm capacity for assets under construction is updated
								
							self.cm_fc_under_construction[j] += cc_inv_total

							# Accumulated investment from capacity market

							self.capacity_cm_uc[i,j] += cc_inv_total

							# Accumulated costs of investment in a particular technology

							self.inv_cost_accum_cm[i,j] += inv_vector[0,5]
						
						else:

							# Storage capacity

							self.cm_fc_under_construction_battery_cm += cc_inv_total

							# Accumulated cm investment

							self.capacity_cm_battery_uc[i] += cc_inv_total

							# Accumulated costs of investment in a particular technology

							self.inv_cost_accum_cm_battery[i] += inv_vector[0,5]

				# Updating 
					
				self.under_construction.update({a:inv_under_construction})

				i += 1

		elif not self.cm_auction_indicator and self.month == 3:

			self.cm_price = 0

		""" Contracts for Difference auctions """

		p_g_CfD = np.zeros([self.agents_g, self.act_bids_CfD])
		q_g_CfD = np.zeros([self.agents_g, self.act_bids_CfD])

		## Capacity market auction (in case auction indicator was activated in the last period)

		if self.CfD_auction_indicator and self.month == 5 and self.CfD_activation:

			# Looping through agents and technologies to get quantity and price bids to the markets

			i = 0

			for a in self.possible_agents_g:

				action_temp[i,:] = actions_g[i]  

				for j in range(self.n_tech_RES):
					
					# Price bids for capacity market
					p_g_CfD[i, j] = (action_temp[i, self.act_inv + self.act_inv_cm + self.act_bids_cm + j] - 1)/(self.step_g_bids - 2) * self.CfD_price_cap
					
					# Quantity bids for capacity market
					q_g_CfD[i, j] = ((action_temp[i, self.act_inv + self.act_inv_cm + self.act_bids_cm + self.act_bids_CfD + j] - 1)/(self.step_g_inv - 2)) * (self.inv_max[j] * self.availability_tech_average_yearly[self.year_int, j] * self.average_failures[0,j])
				
				i += 1

			p_g_CfD = np.where(p_g_CfD < 0, 0, p_g_CfD)

			q_g_CfD = np.where(q_g_CfD < 0, 0, q_g_CfD)

			p_CfD_demand = np.ones(2) * self.CfD_price_cap[0]

			q_CfD_demand = np.ones(2) * max(self.CfD_balance, 0)/2

			p_g_CfD_flatt = p_g_CfD.flatten()

			q_g_CfD_flatt = q_g_CfD.flatten()
						
			CfD_price_flatt, q_accepted_u_CfD, q_accepted_g_CfD_flatt, self.CfD_scarcity = self.double_side_auction_pay_as_bid(p_CfD_demand,
																				  q_CfD_demand, p_g_CfD_flatt, q_g_CfD_flatt, self.random_g_CfD, self.CfD_price_cap)
									
			self.CfD_price = CfD_price_flatt.reshape((self.agents_g, self.act_bids_CfD))

			q_accepted_g_CfD = q_accepted_g_CfD_flatt.reshape((self.agents_g, self.act_bids_CfD))

			i = 0
			
			for a in self.possible_agents_g:

				inv_under_construction = self.under_construction.get(a)

				for j in range(self.n_tech_RES):

					# Vector for new investment (if action was taken to invest). All winning projects are built

					CfD_inv_temp = q_accepted_g_CfD[i,j]
					CfD_inv_total = q_g_CfD[i, j] / (self.availability_tech_average_yearly[self.year_int, j] * self.average_failures[0,j] + self.delta_small)
							
					if CfD_inv_temp > 0 and self.max_year - self.year >  self.construction_time[j]:

						inv_vector = np.zeros([1,6])
						inv_vector[0,0] = j
						inv_vector[0,1] = CfD_inv_total  
						inv_vector[0,2] = self.construction_time[j]
						inv_vector[0,3] = 2
						inv_vector[0,4] = self.CfD_price[i,j]
						inv_vector[0,5] = CfD_inv_total * self.inv_cost[self.year_int, j] / self.normalization_factor

						inv_under_construction = np.vstack([inv_under_construction, inv_vector])

						# Firm capacity for assets under construction is updated
							
						self.cm_fc_under_construction[j] += CfD_inv_total

						# Accumulated investment from capacity market

						self.capacity_CfD_uc[i,j] += CfD_inv_total

						# Accumulated costs of investment in a particular technology

						self.inv_cost_accum[i,j] += inv_vector[0,5]
						self.inv_cost_accum_CfD[i,j] += inv_vector[0,5]

				# Updating 
					
				self.under_construction.update({a:inv_under_construction})

				i += 1
			
		elif not self.CfD_auction_indicator and self.month == 5:

			self.CfD_price = 0

		""" Merchant investments """

		# In case the period corresponds to investments, under constructions assets are checked. Otherwise section is ignored

		i = 0

		m_inv_temp = np.zeros([self.agents_g, self.act_inv])

		if self.month == 1:

			for a in self.possible_agents_g:

				inv_under_construction = self.under_construction.get(a)
					
				action_temp[i,:] = actions_g[i]  

				# Generation assets bids

				for j in range(self.act_inv):
					
					if action_temp[i,j] > 0:

						# Checking if the investment is a generation asset

						if j  < self.n_tech:

							inv_max = self.inv_max[j]

							construction_time = self.construction_time[j]

							inv_cost = self.inv_cost[self.year_int, j]
						
						else:

							inv_max = self.inv_max_battery

							construction_time = self.construction_time_battery

							inv_cost = self.inv_cost_battery[self.year_int]

						# Investment from actions

						m_inv_temp[i, j] = ((action_temp[i,j] - 1)/(self.step_g_inv - 2)) * inv_max

						# Vector for new investment (if action was taken to invest)
							
						if m_inv_temp[i, j] > 0 and self.max_year - self.year >  construction_time:

							inv_vector = np.zeros([1,6])
							inv_vector[0,0] = j
							inv_vector[0,1] = m_inv_temp[i, j]  
							inv_vector[0,2] = construction_time
							inv_vector[0,3] = 0
							inv_vector[0,4] = 0
							inv_vector[0,5] = m_inv_temp[i, j] * inv_cost / self.normalization_factor

							inv_under_construction = np.vstack([inv_under_construction, inv_vector])

							if j  < self.n_tech:

								# Firm capacity for assets under construction is updated

								self.cm_fc_under_construction[j] += m_inv_temp[i, j]

								# Accumulated merchant investment

								self.capacity_merchant_uc[i,j] += m_inv_temp[i, j]

								# Accumulated costs of investment in a particular technology

								self.inv_cost_accum_merchant[i,j] += inv_vector[0,5]

							else:

								self.cm_fc_under_construction_battery_merchant += m_inv_temp[i, j]

								# Accumulated merchant investment

								self.capacity_merchant_battery_uc[i] += m_inv_temp[i, j]

								# Accumulated costs of investment in a particular technology

								self.inv_cost_accum_merchant_battery[i] += inv_vector[0,5]

				# Updating 
						
				self.under_construction.update({a:inv_under_construction})

				i += 1
				
		""" Changing status from assets under construction to in operation """

		i = 0

		for a in self.possible_agents_g:

			rows_to_delete = []	

			# Get the investments under construction and append new investment

			inv_under_construction = self.under_construction.get(a)

			inv_under_construction[:, 2] -= 1/self.yearly_resolution

			k = 0
							
			for vector_under_construction in inv_under_construction:

				if inv_under_construction[k, 2] <= 0 and int(vector_under_construction[0]) < self.n_tech:

					# In case construction has finished, the corresponding investments are added to effective investments (per technology)

					self.inv_g[i, int(vector_under_construction[0])] += vector_under_construction[1]						
						
					# Reducing the firm capacity of assets under construction

					self.cm_fc_under_construction[int(vector_under_construction[0])] -= vector_under_construction[1] 

					# Adding income for new plants coming from the Capacity Mechanism (in monthly resolution)

					if vector_under_construction[3] == 1:

						self.inv_age_cm[i,int(vector_under_construction[0])] = self.inv_age_cm[i,int(vector_under_construction[0])] * self.capacity_cm[i,int(vector_under_construction[0])]/(vector_under_construction[1] + self.capacity_cm[i,int(vector_under_construction[0])] + self.delta_small)

						self.cm_income[i,int(vector_under_construction[0])] += vector_under_construction[1] * vector_under_construction[4] * self.cm_tech_cc[0, int(vector_under_construction[0])] * self.short_t * self.hour_month / self.normalization_factor

						self.cm_inv_options[i, int(vector_under_construction[0])] += vector_under_construction[1] * self.cm_tech_cc[0, int(vector_under_construction[0])] 

						self.capacity_cm[i,int(vector_under_construction[0])] += vector_under_construction[1]

						self.inv_cost_accum_cm[i,int(vector_under_construction[0])] -= vector_under_construction[5]
						
					elif vector_under_construction[3] == 2:

						self.inv_age_CfD[i,int(vector_under_construction[0])] = self.inv_age_CfD[i,int(vector_under_construction[0])] * self.capacity_CfD[i,int(vector_under_construction[0])]/(vector_under_construction[1] + self.capacity_CfD[i,int(vector_under_construction[0])] + self.delta_small)

						self.capacity_CfD[i,int(vector_under_construction[0])] += vector_under_construction[1]

						self.CfD_price_pond[i,int(vector_under_construction[0])] += vector_under_construction[1] * vector_under_construction[4]

						self.CfD_price_agents[i,int(vector_under_construction[0])] = self.CfD_price_pond[i,int(vector_under_construction[0])]/self.capacity_CfD[i,int(vector_under_construction[0])] 

						self.inv_cost_accum_CfD[i,int(vector_under_construction[0])] -= vector_under_construction[5]
						
					elif vector_under_construction[3] == 0:

						self.inv_age_merchant[i,int(vector_under_construction[0])] = self.inv_age_merchant[i,int(vector_under_construction[0])] * self.capacity_merchant[i,int(vector_under_construction[0])]/(vector_under_construction[1] + self.capacity_merchant[i,int(vector_under_construction[0])] + self.delta_small)

						self.capacity_merchant[i,int(vector_under_construction[0])] += vector_under_construction[1]

						self.inv_cost_accum_merchant[i,int(vector_under_construction[0])] -= vector_under_construction[5]

					else:
						raise ValueError(f"Error in resource construction, value in vector {vector_under_construction[3]}")

					rows_to_delete.append(k)

				elif inv_under_construction[k, 2] <= 0 and int(vector_under_construction[0]) == self.n_tech:

					# In case construction has finished, the corresponding investments are added to effective investments (per technology)
					self.inv_g_battery[i] += vector_under_construction[1]

						# Adding income for new plants coming from the Capacity Mechanism (in monthly resolution)
						
					if vector_under_construction[3] == 0:

						self.cm_fc_under_construction_battery_merchant -= vector_under_construction[1] 

						self.inv_age_merchant_battery[i] = self.inv_age_merchant_battery[i] * self.capacity_merchant_battery[i]/(vector_under_construction[1] + self.capacity_merchant_battery[i] + self.delta_small)

						self.capacity_merchant_battery[i] += vector_under_construction[1]

						self.inv_cost_accum_merchant_battery[i] -= vector_under_construction[5]
						
					else:

						self.cm_income_battery[i] += vector_under_construction[1] * vector_under_construction[4] * self.cm_tech_cc[0, int(vector_under_construction[0])] * self.short_t * self.hour_month / self.normalization_factor

						self.cm_inv_options_battery[i] += vector_under_construction[1] * self.cm_tech_cc[0, int(vector_under_construction[0])] 

						self.inv_age_cm_battery[i] = self.inv_age_cm_battery[i] * self.capacity_cm_battery[i]/(vector_under_construction[1] + self.capacity_cm_battery[i] + self.delta_small)

						self.capacity_cm_battery[i] += vector_under_construction[1]

						self.inv_cost_accum_cm_battery[i] -= vector_under_construction[5]

						self.cm_fc_under_construction_battery_cm -= vector_under_construction[1] 

					rows_to_delete.append(k)

				k += 1
						
				# Deleting rows for constructed projects
					
			inv_under_construction = np.delete(inv_under_construction, rows_to_delete, axis=0)

			# Under construction dict
					
			self.under_construction.update({a:inv_under_construction})
					
			i += 1 

		""" Deficit for market calculation """
		
		## Capacity Market - Deficit calculation

		self.cm_balance, self.cm_auction_indicator, self.cm_tech_cc, self.hourly_balance, self.cm_balance_real = self.cm_balance_estimation()

		## CfD - Deficit calculation

		self.CfD_balance, self.CfD_auction_indicator, self.CfD_balance_real = self.CfD_balance_estimation()

		""" Handling Storage stepping """

		## SoC target for storage

		i = 0

		SoC_target = np.zeros([self.agents_g])

		adjusted_inflow = np.zeros([self.agents_g])

		self.adjusted_inflow_charge = np.zeros([self.agents_g])

		self.adjusted_inflow_discharge = np.zeros([self.agents_g])

		self.storage_actions = np.zeros([self.agents_g])

		# Long-term storage

		for a in self.possible_agents_g:

			action_temp[i,:] = actions_g[i]  

			SoC_target[i] = (action_temp[i, self.act_inv + self.act_inv_cm + self.act_bids_cm + self.act_inv_CfD + self.act_bids_CfD])/(self.step_SoC_control - 1) * self.SoC_max_merchant_storage_lt[i] 

			self.storage_actions[i] = action_temp[i, self.act_inv + self.act_inv_cm + self.act_bids_cm + self.act_inv_CfD + self.act_bids_CfD]

			adjusted_inflow[i] = (SoC_target[i] - self.SoC_merchant_storage_lt[i])/(self.capacity_existing_storage_lt[i] * self.short_t * self.hour_month + self.delta_small)

			if adjusted_inflow[i] >= 0:

				self.adjusted_inflow_charge[i] = adjusted_inflow[i]

			else:

				self.adjusted_inflow_discharge[i] = adjusted_inflow[i]

			i += 1

		self.SoC_merchant_storage_lt = SoC_target

		""" Looping for short-term market """

		# Converting actions to nominal values (GENCOs)

		p_g_m = np.zeros([self.agents_g, self.n_tech])

		# Short-term storage bids
		q_u_b = np.zeros([self.agents_g,3])
		p_u_b = np.zeros([self.agents_g,3])

		spot_payment_trimester = 0
		demand_trimester = 0
	
		for t in range (self.short_t):

			spot_payment_hour = 0

			demand_hour = 0

			spot_payment_trimester += np.sum(self.cm_income) * self.normalization_factor/(self.hour_month * self.short_t)

			spot_payment_hour += np.sum(self.cm_income) * self.normalization_factor/(self.hour_month * self.short_t)

			spot_payment_trimester += np.sum(self.cm_income_battery) * self.normalization_factor/(self.hour_month * self.short_t)

			spot_payment_hour += np.sum(self.cm_income_battery) * self.normalization_factor/(self.hour_month * self.short_t)

			i = 0

			p_g = np.zeros([self.agents_g, self.n_tech + 3])
			q_g = np.zeros([self.agents_g, self.n_tech + 3])

			for a in self.possible_agents_g:

				# Generation assets bids

				for j in range(self.n_tech):
					
					# q bids:

					q_g[i,j] = self.inv_g[i, j] * (self.availability_tech_step[t, j]) * self.aggregated_failures[i,j]
					
					# Aggregated variable cost (including carbon tax)
					
					aggregated_vc = (self.v_c_g[self.hour_year, j]) + self.CO2_tax_tech[self.year_int, j]

					p_g[i,j] = (np.double(aggregated_vc))
					p_g_m[i,j] = p_g[i,j]

				# Storage bids for short-term market

				p_g[i, self.n_tech] = 0
				q_g[i, self.n_tech] = self.strategy_discharge_4_hours[t] * self.capacity_merchant_battery[i] 

				p_u_b[i,0] = self.VoLL
				q_u_b[i,0] = self.strategy_charge_4_hours[t] * self.capacity_merchant_battery[i] 

				p_g[i, self.n_tech + 1] = 0
				q_g[i, self.n_tech + 1] = self.strategy_discharge_3_hours[t] * self.capacity_cm_battery[i]
				
				p_u_b[i,1] = self.VoLL
				q_u_b[i,1] = self.strategy_charge_3_hours[t] * self.capacity_cm_battery[i]

				p_g[i, self.n_tech + 2] = 0
				q_g[i, self.n_tech + 2] = (self.strategy_discharge_8_hours[t] + self.adjusted_inflow_discharge[i] + self.availability_tech_step[t, self.n_tech]) * self.capacity_existing_storage_lt[i]

				p_u_b[i,2] = self.VoLL
				q_u_b[i,2] = (self.strategy_charge_8_hours[t] + self.adjusted_inflow_charge[i]) * self.capacity_existing_storage_lt[i]

				i += 1

			# Demand bid utilites (minimum 2 for the auction algorithm to work)
			
			demand_temp = self.demand_step[t] - self.availability_tech_step[t, self.n_tech + 1]

			q_u_non_flex = np.ones(2) * demand_temp/2
			p_u_non_flex = np.ones(2) * self.VoLL

			# Concatenting bids
			
			q_u_b_flatt = q_u_b.flatten()
			p_u_b_flatt = p_u_b.flatten()

			q_u_t = np.concatenate((q_u_b_flatt, q_u_non_flex), axis=0)
			p_u_t = np.concatenate((p_u_b_flatt, p_u_non_flex), axis=0)
			q_g_t = q_g.flatten()
			p_g_t = p_g.flatten()
	
			# Running double-sided auction

			price, q_accepted_u_flatt, q_accepted_g_flatt, LL = self.double_side_auction_marginal_pricing(p_u_t,q_u_t,p_g_t,q_g_t, self.random_g_p, self.VoLL)
			
			q_accepted_g = q_accepted_g_flatt.reshape((self.agents_g, self.n_tech + 3))

			q_accepted_u = q_accepted_u_flatt[:int(self.agents_g * 3)].reshape((self.agents_g, 3))

			self.hourly_prices[t] = price

			spot_payment_trimester += price * sum(q_accepted_g_flatt)
			spot_payment_hour += price * sum(q_accepted_g_flatt)
			
			demand_trimester += sum(q_accepted_u_flatt) 
			demand_hour += sum(q_accepted_u_flatt)

			# Populating observations and rewards (GENCOs)

			i = 0

			for a in self.possible_agents_g:
				
				# Reward from technology 

				for j in range(self.n_tech):
					
					profit_temp_spot = 0

					profit_temp_spot = (((price - 
					  self.v_c_g[self.hour_year, j] - 
					  self.CO2_tax_tech[self.year_int, j]) * 
						q_accepted_g[i,j]) - self.fixed_cost[self.year_int, j] * self.inv_g[i,j])/(self.normalization_factor) * self.hour_month

					# Reward from markets

					self.reward_tech_merchant[i,j] += profit_temp_spot * (self.capacity_merchant[i,j])/(self.capacity_merchant[i,j] + self.capacity_cm[i,j] + self.capacity_CfD[i,j] + self.inv_g_aging[i,j] + self.delta_small) * self.discount_factor_merchant 
					self.reward_tech_merchant_step[i,j] += profit_temp_spot * (self.capacity_merchant[i,j])/(self.capacity_merchant[i,j] + self.capacity_cm[i,j] + self.capacity_CfD[i,j] + self.inv_g_aging[i,j] + self.delta_small) * self.discount_factor_merchant 

					self.reward_tech_existing[i,j] += profit_temp_spot * (self.inv_g_aging[i,j])/(self.capacity_merchant[i,j] + self.capacity_cm[i,j] + self.capacity_CfD[i,j] + self.inv_g_aging[i,j] + self.delta_small) * self.discount_factor 
					self.reward_tech_existing_step[i,j] += profit_temp_spot * (self.inv_g_aging[i,j])/(self.capacity_merchant[i,j] + self.capacity_cm[i,j] + self.capacity_CfD[i,j] + self.inv_g_aging[i,j] + self.delta_small) * self.discount_factor 

					self.reward_tech_cm[i,j] += profit_temp_spot * self.capacity_cm[i,j]/(self.capacity_merchant[i,j] + self.capacity_cm[i,j] + self.capacity_CfD[i,j] + self.inv_g_aging[i,j] + self.delta_small) * self.discount_factor_cm 
					self.reward_tech_cm_step[i,j] += profit_temp_spot * self.capacity_cm[i,j]/(self.capacity_merchant[i,j] + self.capacity_cm[i,j] + self.capacity_CfD[i,j] + self.inv_g_aging[i,j] + self.delta_small) * self.discount_factor_cm 

					self.reward_tech_CfD[i,j] += profit_temp_spot * self.capacity_CfD[i,j]/(self.capacity_merchant[i,j] + self.capacity_cm[i,j] + self.capacity_CfD[i,j] + self.inv_g_aging[i,j] + self.delta_small) * self.discount_factor_CfD 
					self.reward_tech_CfD_step[i,j] += profit_temp_spot * self.capacity_CfD[i,j]/(self.capacity_merchant[i,j] + self.capacity_cm[i,j] + self.capacity_CfD[i,j] + self.inv_g_aging[i,j] + self.delta_small) * self.discount_factor_CfD 

					# Capacity Market Reliability option

					cm_option_value = np.maximum(price - self.cm_option_strike, 0) * self.cm_inv_options[i,j] /(self.normalization_factor) * self.hour_month

					spot_payment_trimester -= np.maximum(price - self.cm_option_strike, 0) * self.cm_inv_options[i,j]
					spot_payment_hour -= np.maximum(price - self.cm_option_strike, 0) * self.cm_inv_options[i,j]

					# Contracts for Difference settlement

					CfD_option_value = (price - self.CfD_price_agents[i,j]) * q_accepted_g[i,j] * self.capacity_CfD[i,j]/(self.capacity_merchant[i,j] + self.capacity_cm[i,j] + self.capacity_CfD[i,j] + self.inv_g_aging[i,j] + self.delta_small)/self.normalization_factor * self.hour_month

					spot_payment_trimester -= (price - self.CfD_price_agents[i,j]) * q_accepted_g[i,j] * self.capacity_CfD[i,j]/(self.capacity_merchant[i,j] + self.capacity_cm[i,j] + self.capacity_CfD[i,j] + self.inv_g_aging[i,j] + self.delta_small)
					spot_payment_hour -= (price - self.CfD_price_agents[i,j]) * q_accepted_g[i,j] * self.capacity_CfD[i,j]/(self.capacity_merchant[i,j] + self.capacity_cm[i,j] + self.capacity_CfD[i,j] + self.inv_g_aging[i,j] + self.delta_small)

					# Emissions and production

					self.CO2_emissions_step += q_accepted_g[i,j] * self.CO2_tax_tech_original[self.year_int, j] / self.CO2_tax_original[self.year_int] * self.hour_month
					
					# Option rewards

					self.reward_tech_cm[i,j] -= cm_option_value * self.discount_factor_cm 
					self.reward_tech_cm_step[i,j] -= cm_option_value * self.discount_factor_cm 

					self.reward_tech_CfD[i,j] -= CfD_option_value * self.discount_factor_CfD 
					self.reward_tech_CfD_step[i,j] -= CfD_option_value * self.discount_factor_CfD 

					# Accumulated profits for termination conditions

					if self.year > self.max_year - self.max_year/4:

						self.accumulated_profit_merchant[j] += profit_temp_spot * (self.capacity_merchant[i,j])/(self.capacity_merchant[i,j] + self.capacity_cm[i,j] + self.capacity_CfD[i,j] + self.inv_g_aging[i,j] + self.delta_small)
						
						self.accumulated_profit_cm[j] += profit_temp_spot * (self.capacity_cm[i,j])/(self.capacity_merchant[i,j] + self.capacity_cm[i,j] + self.capacity_CfD[i,j] + self.inv_g_aging[i,j] + self.delta_small)

						self.accumulated_profit_cm[j] -= cm_option_value 

						self.accumulated_profit_CfD[j] += profit_temp_spot * (self.capacity_CfD[i,j])/(self.capacity_merchant[i,j] + self.capacity_cm[i,j] + self.capacity_CfD[i,j] + self.inv_g_aging[i,j] + self.delta_small)

						self.accumulated_profit_CfD[j] -= CfD_option_value

				# Batteries
				
				income_storage_merchant = (price * (
					-q_accepted_u[i,0] + q_accepted_g[i, self.n_tech]) - self.fixed_cost_battery[self.year_int] * self.capacity_merchant_battery[i])/(self.normalization_factor) * self.hour_month 	
				
				income_storage_cm = (price * (
					-q_accepted_u[i,1] + q_accepted_g[i, self.n_tech + 1]) - self.fixed_cost_battery[self.year_int] * self.capacity_cm_battery[i])/(self.normalization_factor) * self.hour_month 	

				income_storage_lt = (price * (
					-q_accepted_u[i,2] + q_accepted_g[i, self.n_tech + 2]) - self.fixed_cost_storage_lt[self.year_int] * self.capacity_existing_storage_lt[i])/(self.normalization_factor) * self.hour_month 	

				# Reward across markets
				
				self.reward_tech_merchant_battery[i] += income_storage_merchant * self.discount_factor_merchant 
				self.reward_tech_merchant_battery_step[i] += income_storage_merchant * self.discount_factor_merchant 

				self.reward_tech_cm_battery[i] += income_storage_cm * self.discount_factor_cm 
				self.reward_tech_cm_battery_step[i] += income_storage_cm * self.discount_factor_cm 

				self.reward_tech_existing_storage_lt[i] += income_storage_lt * self.discount_factor_cm 
				self.reward_tech_existing_storage_lt_step[i] += income_storage_lt * self.discount_factor_cm 

				cm_option_value_battery = np.maximum(price - self.cm_option_strike, 0) * self.cm_inv_options_battery[i] /(self.normalization_factor) * self.hour_month

				spot_payment_trimester -= np.maximum(price - self.cm_option_strike, 0) * self.cm_inv_options_battery[i]
				spot_payment_hour -= np.maximum(price - self.cm_option_strike, 0) * self.cm_inv_options_battery[i]

				self.reward_tech_cm_battery[i] -= cm_option_value_battery * self.discount_factor_cm 
				self.reward_tech_cm_battery_step[i] -= cm_option_value_battery * self.discount_factor_cm 

				# Accumulated profits for termination conditions

				if self.year > self.max_year - self.max_year/4:

					self.accumulated_profit_merchant_battery += income_storage_merchant * (self.capacity_merchant_battery[i] + self.inv_init_battery[i]) /(self.capacity_merchant_battery[i] + self.capacity_cm_battery[i] + self.inv_init_battery[i] + self.delta_small)
					self.accumulated_profit_cm_battery += income_storage_cm * self.capacity_cm_battery[i] /(self.capacity_merchant_battery[i] + self.capacity_cm_battery[i] + self.inv_init_battery[i] + self.delta_small)

					self.accumulated_profit_cm_battery -= cm_option_value_battery
				
				# Short-term prices for observations

				obs_temp[i,t] = np.clip(((price * 2)/(self.VoLL) - 1), -1, 1)
				
				i += 1

			# Updating time	
			self.time += 1
			self.hour_year += 1
			self.time_random += 1

		# Updating counter for random matrix

		if self.time_random >= 503:

			self.time_random = 0

		""" Resource availability pre-observation """

		# Demand projection

		average_demand_projection_short = self.demand_average_bimester[int(self.year_int * self.yearly_resolution + self.month)]

		average_demand_projection_long = self.demand_average_year[self.year_int + self.planning_horizon]
		
		# Resource availability (short)

		average_solar_availability_short = self.availability_tech_average_bimester[int(self.year_int * self.yearly_resolution + self.month), 0]

		average_wind_availability_short = self.availability_tech_average_bimester[int(self.year_int * self.yearly_resolution + self.month), 1]

		average_hydro_availability_short = self.availability_tech_average_bimester[int(self.year_int * self.yearly_resolution + self.month), self.n_tech]

		average_hydro_ror_availability_short = self.availability_tech_average_bimester[int(self.year_int * self.yearly_resolution + self.month), self.n_tech + 1]

		average_solar_availability_long = self.availability_tech_average_yearly[self.year_int, 0]

		average_wind_availability_long = self.availability_tech_average_yearly[self.year_int, 1]

		average_hydro_availability_long = self.availability_tech_average_yearly[self.year_int, self.n_tech]

		average_hydro_ror_availability_long = self.availability_tech_average_yearly[self.year_int, self.n_tech + 1]

		""" Looping agents for observation and reward handling """
		
		i = 0

		for a in self.possible_agents_g:

			## Resource availability

			# Solar availability 
			obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific) + 0] = np.clip((average_solar_availability_short * 2) - 1, -1, 1)
					
			# Wind availability
			obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific) + 1] = np.clip((average_wind_availability_short * 2) - 1, -1, 1)
					
			# Hydro availability
			obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific) + 2] = np.clip((average_hydro_availability_short * 2) - 1, -1, 1)

			# Solar availability 
			obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific) + 3] = np.clip((average_solar_availability_long * 2) - 1, -1, 1)
					
			# Wind availability
			obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific) + 4] = np.clip((average_wind_availability_long * 2) - 1, -1, 1)
					
			# Hydro availability
			obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific) + 5] = np.clip((average_hydro_availability_long * 2) - 1, -1, 1)

			# Demand projection (short)
			obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific) + 6] = np.clip((average_demand_projection_short/self.max_demand * 2) - 1, -1, 1)

			# Demand projection (short)
			obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific) + 7] = np.clip((average_demand_projection_long/self.max_demand * 2) - 1, -1, 1)

			# Hydro ROR with respect to Demand projection (short)
			obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific) + 8] = np.clip((average_demand_projection_short - average_hydro_ror_availability_short)/(average_demand_projection_short + average_hydro_ror_availability_short), -1, 1)

			# Hydro ROR with respect to Demand projection (long)
			obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific) + 9] = np.clip((average_demand_projection_long - average_hydro_ror_availability_long)/(average_demand_projection_long + average_hydro_ror_availability_long), -1, 1)

			## Installed capacity (individual and total)

			for j in range(self.n_tech):

				ind_installed_capacity = self.inv_g[i,j]

				ind_total_installed_capacity = np.sum(self.inv_g[i,:])

				total_installed_capacity = np.sum(self.inv_g[:,j])

				ind_installed_capacity_existing = self.inv_g_aging[i,j]

				total_installed_capacity_existing = np.sum(self.inv_g_aging[:,j])

				ind_installed_capacity_merchant = self.capacity_merchant_uc[i,j]

				total_installed_capacity_merchant = np.sum(self.capacity_merchant_uc[:,j])

				ind_installed_capacity_cm = self.capacity_cm_uc[i,j]

				total_installed_capacity_cm = np.sum(self.capacity_cm_uc[:,j])

				ind_installed_capacity_CfD = self.capacity_CfD_uc[i,j]

				total_installed_capacity_CfD = np.sum(self.capacity_CfD_uc[:,j])

				## Demand projections with respect to install technologies

				# Individual	
						
				obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific)  + self.n_observation_resources + (self.n_observation_specific_2 * j) + 0] = np.clip((average_demand_projection_long/self.agents_g - ind_installed_capacity)/(average_demand_projection_long/self.agents_g + ind_installed_capacity), -1, 1)

				# Individual	
						
				obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific)  + self.n_observation_resources + (self.n_observation_specific_2 * j) + 1] = np.clip((average_demand_projection_long - total_installed_capacity)/(average_demand_projection_long + total_installed_capacity), -1, 1)
				
				# Individual	
						
				obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific)  + self.n_observation_resources + (self.n_observation_specific_2 * j) + 2] = np.clip((ind_installed_capacity)/(ind_total_installed_capacity + self.delta_small) * 2 - 1, -1, 1)

				# Total

				obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific)  + self.n_observation_resources + (self.n_observation_specific_2 * j) + 3] = np.clip((ind_installed_capacity)/(total_installed_capacity + self.delta_small) * 2 - 1, -1, 1)

				# ind - Existing

				obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific)  + self.n_observation_resources + (self.n_observation_specific_2 * j) + 4] = np.clip(ind_installed_capacity_existing / (ind_installed_capacity + self.delta_small) * 2 - 1, -1, 1)

				# total - existing

				obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific)  + self.n_observation_resources + (self.n_observation_specific_2 * j) + 5] = np.clip((total_installed_capacity_existing)/(total_installed_capacity + self.delta_small) * 2 - 1, -1, 1)

				# ind - Merchant

				obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific)  + self.n_observation_resources + (self.n_observation_specific_2 * j) + 6] = np.clip(ind_installed_capacity_merchant / (ind_installed_capacity + self.delta_small) * 2 - 1, -1, 1)

				# total - Merchant

				obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific)  + self.n_observation_resources + (self.n_observation_specific_2 * j) + 7] = np.clip((total_installed_capacity_merchant)/(total_installed_capacity + self.delta_small) * 2 - 1, -1, 1)

				# ind - CM

				obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific)  + self.n_observation_resources + (self.n_observation_specific_2 * j) + 8] = np.clip(ind_installed_capacity_cm / (ind_installed_capacity + self.delta_small) * 2 - 1, -1, 1)

				# total - CM

				obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific)  + self.n_observation_resources + (self.n_observation_specific_2 * j) + 9] = np.clip((total_installed_capacity_cm)/(total_installed_capacity + self.delta_small) * 2 - 1, -1, 1)

				# ind - CfD

				obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific)  + self.n_observation_resources + (self.n_observation_specific_2 * j) + 10] = np.clip(ind_installed_capacity_CfD / (ind_installed_capacity + self.delta_small) * 2 - 1, -1, 1)

				# total - CfD

				obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific)  + self.n_observation_resources + (self.n_observation_specific_2 * j) + 11] = np.clip((total_installed_capacity_CfD)/(total_installed_capacity + self.delta_small) * 2 - 1, -1, 1)

				# Capacity credits - CM

				obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific)  + self.n_observation_resources + (self.n_observation_specific_2 * j) + 12] = np.clip((self.cm_tech_cc[0,j]) * 2 - 1, -1, 1)

				## Adjustment of rewards due to nvestment costs (Generation)

				self.reward_tech_merchant[i,j] -= self.inv_cost_accum_merchant[i,j] * self.discount_factor_merchant 
				self.reward_tech_merchant_step[i,j] -= self.inv_cost_accum_merchant[i,j] * self.discount_factor_merchant 

				self.reward_tech_cm[i,j] -= self.inv_cost_accum_cm[i,j] * self.discount_factor_cm 
				self.reward_tech_cm_step[i,j] -= self.inv_cost_accum_cm[i,j] * self.discount_factor_cm 

				self.reward_tech_CfD[i,j] -= self.inv_cost_accum_CfD[i,j] * self.discount_factor_CfD 
				self.reward_tech_CfD_step[i,j] -= self.inv_cost_accum_CfD[i,j] * self.discount_factor_CfD 

				## Including income from Capacity Market
					
				self.reward_tech_cm[i,j] += self.cm_income[i,j] * self.discount_factor_cm 
				self.reward_tech_cm_step[i,j] += self.cm_income[i,j] * self.discount_factor_cm 				

				## Accumulated profit 

				if self.year > self.max_year - self.max_year/4:

					self.accumulated_profit_cm[j] += self.cm_income[i,j]
					self.accumulated_profit_cm_battery += self.cm_income_battery[i]

			## Adjustment of rewards due to investment costs (Batteries)

			self.reward_tech_cm_battery[i] += self.cm_income_battery[i] * self.discount_factor_cm 
			self.reward_tech_cm_battery_step[i] += self.cm_income_battery[i] * self.discount_factor_cm 

			self.reward_tech_merchant_battery[i] -= self.inv_cost_accum_merchant_battery[i] * self.discount_factor_merchant 
			self.reward_tech_merchant_battery_step[i] -= self.inv_cost_accum_merchant_battery[i] * self.discount_factor_merchant 

			self.reward_tech_cm_battery[i] -= self.inv_cost_accum_cm_battery[i] * self.discount_factor_cm
			self.reward_tech_cm_battery_step[i] -= self.inv_cost_accum_cm_battery[i] * self.discount_factor_cm 

			## Battery and long-term storage observations

			ind_installed_capacity_merchant_battery = self.capacity_merchant_battery[i]

			total_installed_capacity_merchant_battery = np.sum(self.capacity_merchant_battery)

			ind_installed_capacity_cm_battery = self.capacity_cm_battery[i]

			total_installed_capacity_cm_battery = np.sum(self.capacity_cm_battery)

			ind_installed_capacity_existing_storage_lt = self.capacity_existing_storage_lt[i]

			total_installed_capacity_existing_storage_lt = np.sum(self.capacity_existing_storage_lt)

			ind_SoC_existing_storage_lt = self.SoC_merchant_storage_lt[i]/(self.SoC_max_merchant_storage_lt[i] + self.delta_small)

			total_SoC_existing_storage_lt = np.sum(self.SoC_merchant_storage_lt)/(np.sum(self.SoC_max_merchant_storage_lt) + self.delta_small)

			# Individual - merchant
						
			obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific) + self.n_observation_resources + (self.n_observation_specific_2 * self.n_tech) + 0] = np.clip((average_demand_projection_long/self.agents_g - ind_installed_capacity_merchant_battery)/(average_demand_projection_long/self.agents_g + ind_installed_capacity_merchant_battery), -1, 1)

			# Total	- merchant
						
			obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific) + self.n_observation_resources + (self.n_observation_specific_2 * self.n_tech) + 1]= np.clip((average_demand_projection_long - total_installed_capacity_merchant_battery)/(average_demand_projection_long + total_installed_capacity_merchant_battery), -1, 1)

			# Individual - cm
						
			obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific) + self.n_observation_resources + (self.n_observation_specific_2 * self.n_tech) + 2] = np.clip((average_demand_projection_long/self.agents_g - ind_installed_capacity_cm_battery)/(average_demand_projection_long/self.agents_g + ind_installed_capacity_cm_battery), -1, 1)

			# Total	- merchant
						
			obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific) + self.n_observation_resources + (self.n_observation_specific_2 * self.n_tech) + 3]= np.clip((average_demand_projection_long - total_installed_capacity_cm_battery)/(average_demand_projection_long + total_installed_capacity_cm_battery), -1, 1)

			# Capacity factor battery 

			obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific) + self.n_observation_resources + (self.n_observation_specific_2 * self.n_tech) + 4]= np.clip((self.cm_tech_cc[0,self.n_tech]) * 2 - 1, -1, 1)

			# Individual - Existing Merchant Long-term
						
			obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific) + self.n_observation_resources + (self.n_observation_specific_2 * self.n_tech) + 5] = np.clip((average_demand_projection_long/self.agents_g - ind_installed_capacity_existing_storage_lt)/(average_demand_projection_long/self.agents_g + ind_installed_capacity_existing_storage_lt), -1, 1)

			# Total	- Existing Merchant Long-term
						
			obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific) + self.n_observation_resources + (self.n_observation_specific_2 * self.n_tech) + 6]= np.clip((average_demand_projection_long - total_installed_capacity_existing_storage_lt)/(average_demand_projection_long + total_installed_capacity_existing_storage_lt), -1, 1)

			# Individual - SoC Existing Merchant Long-term
						
			obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific) + self.n_observation_resources + (self.n_observation_specific_2 * self.n_tech) + 7] = np.clip(ind_SoC_existing_storage_lt * 2 - 1, -1, 1)

			# Total	- SoC Existing Merchant Long-term
						
			obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific) + self.n_observation_resources + (self.n_observation_specific_2 * self.n_tech) + 8]= np.clip(total_SoC_existing_storage_lt * 2 - 1, -1, 1)

			## Time observations

			# Month

			obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific) + self.n_observation_resources + (self.n_observation_specific_2 * self.n_tech) + self.n_observation_specific_2_battery + 0] = np.clip((self.month/self.yearly_resolution) * 2 - 1, -1, 1)
   
			# Year
   
			obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific) + self.n_observation_resources + (self.n_observation_specific_2 * self.n_tech) + self.n_observation_specific_2_battery + 1] = np.clip((self.year/self.max_year) * 2 - 1, -1, 1)
   
			# Time
   
			obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific) + self.n_observation_resources + (self.n_observation_specific_2 * self.n_tech) + self.n_observation_specific_2_battery + 2] = np.clip(self.hour_year/(self.max_year * self.yearly_resolution) * 2 - 1, -1, 1)

			# CO2 tax
   
			obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific) + self.n_observation_resources + (self.n_observation_specific_2 * self.n_tech) + self.n_observation_specific_2_battery + 3] = np.clip((self.CO2_tax[self.year_int]/300) * 2 - 1, -1, 1)
			
			## Capacity Market and CfD observations

			# CM Balance 

			obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific) + self.n_observation_resources + (self.n_observation_specific_2 * self.n_tech) + self.n_observation_specific_2_battery + self.n_observation_time + 0] = np.clip(self.cm_balance_real/self.percentile_demand,-1,1)

			# CM price

			obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific) + self.n_observation_resources + (self.n_observation_specific_2 * self.n_tech) + self.n_observation_specific_2_battery + self.n_observation_time + 1] = np.clip(((((np.sum(self.cm_income) + np.sum(self.cm_income_battery)) * self.normalization_factor /(self.short_t * self.hour_month))/(np.sum(self.cm_inv_options) + np.sum(self.cm_inv_options_battery) + self.delta_small))/self.cm_price_cap) * 2 - 1, -1, 1)

			# CM scarcity

			obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific) + self.n_observation_resources + (self.n_observation_specific_2 * self.n_tech) + self.n_observation_specific_2_battery + self.n_observation_time + 2] = np.clip((self.cm_scarcity * 2) - 1, -1, 1)

			# CfD Balance 

			obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific) + self.n_observation_resources + (self.n_observation_specific_2 * self.n_tech) + self.n_observation_specific_2_battery + self.n_observation_time + 3] = np.clip(self.CfD_balance_real/self.percentile_demand,-1,1)

			# CfD Price 

			obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific) + self.n_observation_resources + (self.n_observation_specific_2 * self.n_tech) + self.n_observation_specific_2_battery + self.n_observation_time + 4] = np.clip(((np.sum(self.CfD_price_pond)/(np.sum(self.capacity_CfD) + self.delta_small))/self.CfD_price_cap) * 2 - 1, -1, 1)

			# CfD scarcity

			obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific) + self.n_observation_resources + (self.n_observation_specific_2 * self.n_tech) + self.n_observation_specific_2_battery + self.n_observation_time + 5] = np.clip((self.CfD_scarcity * 2) - 1, -1, 1)

			""" Termination conditions """

			if self.time >= self.max_t: 

				for j in range(self.n_tech):
					
					npv_annuaity_MW_merchant = (self.accumulated_profit_merchant[j] / (np.sum(self.capacity_merchant[:,j]) + self.delta_small)) * ((1 - (1 + self.opportunity_cost_merchant)**(-(np.max(self.life_time[j] - int(self.inv_age_merchant[i,j]),0))))/(self.opportunity_cost_merchant))

					npv_annuaity_MW_cm = (self.accumulated_profit_cm[j] / (np.sum(self.capacity_cm[:,j]) + self.delta_small)) * ((1 - (1 + self.opportunity_cost_cm)**(-(np.max(self.life_time[j] - int(self.inv_age_cm[i,j]),0))))/(self.opportunity_cost_cm))

					npv_annuaity_MW_CfD = (self.accumulated_profit_CfD[j] / (np.sum(self.capacity_CfD[:,j]) + self.delta_small)) * ((1 - (1 + self.opportunity_cost_CfD)**(-(np.max(self.life_time[j] - int(self.inv_age_CfD[i,j]),0))))/(self.opportunity_cost_CfD))
					

					depreciation_value_merchant = npv_annuaity_MW_merchant * (self.capacity_merchant[i,j])/(self.count_last[j])

					depreciation_value_cm = npv_annuaity_MW_cm * (self.capacity_cm[i,j])/(self.count_last[j]) 

					depreciation_value_CfD = npv_annuaity_MW_CfD * (self.capacity_CfD[i,j])/(self.count_last[j]) 

					# Accumulating reward

					self.reward_tech_merchant[i,j] += depreciation_value_merchant * self.discount_factor_merchant 
					self.reward_tech_merchant_step[i,j] += depreciation_value_merchant * self.discount_factor_merchant 

					self.reward_tech_cm[i,j] += depreciation_value_cm * self.discount_factor_cm
					self.reward_tech_cm_step[i,j] += depreciation_value_cm * self.discount_factor_cm

					self.reward_tech_CfD[i,j] += depreciation_value_CfD * self.discount_factor_CfD
					self.reward_tech_CfD_step[i,j] += depreciation_value_CfD * self.discount_factor_CfD
						
			## Depreciation value battery

			if self.time >= self.max_t: 
					
				npv_annuaity_MW_merchant_battery = (self.accumulated_profit_merchant_battery / (np.sum(self.capacity_merchant_battery) + self.delta_small)) * ((1 - (1 + self.opportunity_cost_merchant)**(-(np.max(self.life_time_battery - int(self.inv_age_merchant_battery[i]),0))))/(self.opportunity_cost_merchant))
				depreciation_value_merchant_battery = npv_annuaity_MW_merchant_battery * (self.capacity_merchant_battery[i])/(self.count_last_battery)

				self.reward_tech_merchant_battery[i] += depreciation_value_merchant_battery * self.discount_factor_merchant 
				self.reward_tech_merchant_battery_step[i] += depreciation_value_merchant_battery * self.discount_factor_merchant 
					
				npv_annuaity_MW_cm_battery = (self.accumulated_profit_cm_battery / (np.sum(self.capacity_cm_battery) + self.delta_small)) * ((1 - (1 + self.opportunity_cost_cm)**(-(np.max(self.life_time_battery - int(self.inv_age_cm_battery[i]),0))))/(self.opportunity_cost_cm))
				depreciation_value_cm_battery = npv_annuaity_MW_cm_battery * (self.capacity_cm_battery[i])/(self.count_last_battery)

				self.reward_tech_cm_battery[i] += depreciation_value_cm_battery * self.discount_factor_cm 
				self.reward_tech_cm_battery_step[i] += depreciation_value_cm_battery * self.discount_factor_cm

			for j in range(self.n_tech):

				## Reward observations 

				obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific) + self.n_observation_resources + (self.n_observation_specific_2 * self.n_tech) + self.n_observation_specific_2_battery + self.n_observation_time + self.n_observation_cm_CfD + j * self.n_observation_reward_tech + 0] = np.clip(self.reward_tech_merchant[i,j], -1, 1)
				
				obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific) + self.n_observation_resources + (self.n_observation_specific_2 * self.n_tech) + self.n_observation_specific_2_battery + self.n_observation_time + self.n_observation_cm_CfD + j * self.n_observation_reward_tech + 1] = np.clip(self.reward_tech_cm[i,j], -1, 1)

				obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific) + self.n_observation_resources + (self.n_observation_specific_2 * self.n_tech) + self.n_observation_specific_2_battery + self.n_observation_time + self.n_observation_cm_CfD + j * self.n_observation_reward_tech + 2] = np.clip(self.reward_tech_existing[i,j], -1, 1)

				# Specfic price per technology

				obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific) + self.n_observation_resources + (self.n_observation_specific_2 * self.n_tech) + self.n_observation_specific_2_battery + self.n_observation_time + self.n_observation_cm_CfD + j * self.n_observation_reward_tech + 5] = np.clip(((self.cm_income[i,j] * self.normalization_factor /((self.short_t * self.hour_month)) / (self.capacity_cm[i,j] + self.delta_small)) / self.cm_price_cap) * 2 - 1, -1, 1)

			for j in range(self.n_tech_RES):

				# Reward observations
				
				obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific) + self.n_observation_resources + (self.n_observation_specific_2 * self.n_tech) + self.n_observation_specific_2_battery + self.n_observation_time + self.n_observation_cm_CfD + self.n_tech * self.n_observation_reward_tech + j * self.n_observation_reward_tech_CfD + 0] = np.clip(self.reward_tech_CfD[i,j], -1, 1)

				# Specific price per technology

				obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific) + self.n_observation_resources + (self.n_observation_specific_2 * self.n_tech) + self.n_observation_specific_2_battery + self.n_observation_time + self.n_observation_cm_CfD + self.n_tech * self.n_observation_reward_tech + j * self.n_observation_reward_tech_CfD + 1] = np.clip((self.CfD_price_agents[i,j] / self.CfD_price_cap) * 2 - 1, -1, 1)

			# Reward observation batteries

			obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific) + self.n_observation_resources + (self.n_observation_specific_2 * self.n_tech) + self.n_observation_specific_2_battery + self.n_observation_time + self.n_observation_cm_CfD + self.n_tech * self.n_observation_reward_tech + self.n_tech_RES * self.n_observation_reward_tech_CfD + 0] = np.clip(self.reward_tech_merchant_battery[i], -1, 1)

			obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific) + self.n_observation_resources + (self.n_observation_specific_2 * self.n_tech) + self.n_observation_specific_2_battery + self.n_observation_time + self.n_observation_cm_CfD + self.n_tech * self.n_observation_reward_tech + self.n_tech_RES * self.n_observation_reward_tech_CfD + 1] = np.clip(self.reward_tech_cm_battery[i], -1, 1)

			obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific) + self.n_observation_resources + (self.n_observation_specific_2 * self.n_tech) + self.n_observation_specific_2_battery + self.n_observation_time + self.n_observation_cm_CfD + self.n_tech * self.n_observation_reward_tech + self.n_tech_RES * self.n_observation_reward_tech_CfD + 2] = np.clip(((self.cm_income_battery[i] * self.normalization_factor /((self.short_t * self.hour_month)) / (self.capacity_cm_battery[i] + self.delta_small)) / self.cm_price_cap) * 2 - 1, -1, 1)

			obs_temp[i, self.n_observations_general + (self.n_tech * self.n_observation_specific) + self.n_observation_resources + (self.n_observation_specific_2 * self.n_tech) + self.n_observation_specific_2_battery + self.n_observation_time + self.n_observation_cm_CfD + self.n_tech * self.n_observation_reward_tech + self.n_tech_RES * self.n_observation_reward_tech_CfD + 3] = np.clip(self.reward_tech_existing_storage_lt_step[i], -1, 1)

			observations_temp = {"observations": np.float16(obs_temp[i,:]),
									"action_mask":self.get_action_masks_g(self.investments_enabled[i,:], self.capacity_existing_storage_lt[i])} 
			
			observations_g.update({a:observations_temp})
			
			""" Summing rewards """

			# Accumulating reward

			reward_temp[i] += np.sum(self.reward_tech_existing_step[i,:] + self.reward_tech_merchant_step[i,:] + self.reward_tech_cm_step[i,:] + self.reward_tech_CfD_step[i,:]) + self.reward_tech_merchant_battery_step[i] + self.reward_tech_cm_battery_step[i] + self.reward_tech_existing_storage_lt_step[i]

			self.reward_total = reward_temp

			rewards_g.update({a:reward_temp[i]})
			
			i += 1

		self.price_net = spot_payment_trimester/demand_trimester

		""" Termination conditions """
		
		##  Termination conditions for the Environment

		if self.time >= self.max_t: 
			
			i = 0
			
			for a in self.possible_agents:
				
				self.terminateds.add(a)
				self.truncateds.add(a) 

				i += 1

			terminated = {a: True for a in self.possible_agents}
			truncated = {a: True for a in self.possible_agents}

		observations = observations_g
		rewards = rewards_g

		# Counting years and months
		for j in range(self.n_tech):
			if self.year > self.max_year - self.max_year/4:
				self.count_last[j] += 1/self.yearly_resolution

		if self.year > self.max_year - self.max_year/4:
			self.count_last_battery += 1/self.yearly_resolution

		self.year += 1/self.yearly_resolution

		self.month += 1

		""" Adjusting asset ages """

		if self.month >= self.yearly_resolution:

			self.month = 0
			self.hour_year = 0

			# Aging update for existing assets

			self.inv_age[self.inv_new > 0] += 1

			self.inv_age_cm[self.capacity_cm > 0] += 1

			self.inv_age_CfD[self.capacity_CfD > 0] += 1

			self.inv_age_merchant[self.capacity_merchant > 0] += 1

			self.inv_age_merchant_battery[self.capacity_merchant_battery > 0] += 1

			self.inv_age_cm_battery[self.capacity_cm_battery > 0] += 1

			# Aging update for existing assets

			for j in range(self.n_tech):

				self.inv_g[:,j] -= self.inv_init_g[:,j] * self.aging[self.year_int , j]
				self.inv_g_aging[:,j] -= self.inv_init_g[:,j] * self.aging[self.year_int , j]
			
			# Avoiding zeros
			self.inv_g =np.maximum(self.inv_g, 0)
			self.inv_g [self.inv_g  < 1] = 0
			self.inv_g_aging = np.maximum(self.inv_g_aging, 0)
			self.inv_g_aging [self.inv_g_aging  < 1] = 0

		# Eliminating agents after termination/truncation (for RLLIB compatibility)
		terminated["__all__"] = len(self.terminateds) == len(self.agents)
		truncated["__all__"] = len(self.truncateds) == len(self.agents)

		return observations, rewards, terminated, truncated, info
	
	""" Double-sided auction - Marginal Pricing """
	
	def double_side_auction_marginal_pricing(self, price_u, q_u_o, price_g, q_g_o, random, price_cap):
		
		total_demand = np.sum(q_u_o)
		total_supply = np.sum(q_g_o)

		q_g = copy.deepcopy(q_g_o)
		q_u = copy.deepcopy(q_u_o)
		
		if total_supply >= total_demand:
			
			LL = False  # No load loss

            # Step 2: Sort buyers by descending price (high to low)
			sorted_buy_indices = np.argsort(-price_u)
			sorted_price_u = price_u[sorted_buy_indices]
			sorted_q_u = q_u[sorted_buy_indices]

			sorted_sell_indices = np.lexsort((random[int(self.time_random % 503), :], price_g))
			sorted_price_g = price_g[sorted_sell_indices]
			sorted_q_g = q_g[sorted_sell_indices]

            # Initialize variables for matching
			buyer_idx = 0
			seller_idx = 0
			final_price = 0

            # Step 4: Matching buyers and sellers
			while buyer_idx < len(sorted_price_u) and seller_idx < len(sorted_price_g):
                # Get the buyer and seller's prices
				buyer_price = sorted_price_u[buyer_idx]
				seller_price = sorted_price_g[seller_idx]

                # Check if buyer is willing to pay more or equal to seller's price
				if buyer_price >= seller_price:
                    # Determine the quantity to be matched
					matched_quantity = min(sorted_q_u[buyer_idx], sorted_q_g[seller_idx])

                    # Reduce quantities of both buyer and seller
					sorted_q_u[buyer_idx] -= matched_quantity
					sorted_q_g[seller_idx] -= matched_quantity

                    # Update the final transaction price as seller's price
					final_price = seller_price

                    # Move to the next buyer if their quantity is fully matched
					if sorted_q_u[buyer_idx] == 0:
						buyer_idx += 1

                    # Move to the next seller if their quantity is fully matched
					if sorted_q_g[seller_idx] == 0:
						seller_idx += 1
				else:
                    # If no match is possible, stop the auction
					break

            # Final accepted quantities (already adjusted in sorted_q_u and sorted_q_g)
			q_accepted_u = np.zeros_like(q_u)
			q_accepted_u[sorted_buy_indices] = q_u[sorted_buy_indices] - sorted_q_u

			q_accepted_g = np.zeros_like(q_g)
			q_accepted_g[sorted_sell_indices] = q_g[sorted_sell_indices] - sorted_q_g

		else:
            # Case where total supply is insufficient to meet demand
			
			LL = True  # Load loss
			
			final_price = price_cap  # Set price to cap

            # Accept all seller quantities
			q_accepted_g = q_g_o

            # Adjust buyer quantities proportionally to available supply
			q_accepted_u = q_u * total_supply / total_demand

		return final_price, q_accepted_u, q_accepted_g, LL
	
	""" Double-sided auction - Pay as Bid """

	def double_side_auction_pay_as_bid(self, price_u, q_u_o, price_g, q_g_o, random, price_cap):
		
		total_demand = np.sum(q_u_o)
		total_supply = np.sum(q_g_o)

		q_g = copy.deepcopy(q_g_o)
		q_u = copy.deepcopy(q_u_o)
		
		if total_supply >= total_demand:
			
			LL = False  # No load loss

            # Step 2: Sort buyers by descending price (high to low)
			sorted_buy_indices = np.argsort(-price_u)
			sorted_price_u = price_u[sorted_buy_indices]
			sorted_q_u = q_u[sorted_buy_indices]

			sorted_sell_indices = np.lexsort((random[int(self.time_random % 503), :], price_g))
			sorted_price_g = price_g[sorted_sell_indices]
			sorted_q_g = q_g[sorted_sell_indices]

            # Initialize variables for matching
			buyer_idx = 0
			seller_idx = 0
			final_price = 0

			# Initialize variables for matching
			buyer_idx = 0
			seller_idx = 0
			final_prices_sellers = np.zeros_like(price_g)  # To store final prices paid to sellers

			# Step 4: Matching buyers and sellers
			while buyer_idx < len(sorted_price_u) and seller_idx < len(sorted_price_g):
				# Get the buyer and seller's prices
				buyer_price = sorted_price_u[buyer_idx]
				seller_price = sorted_price_g[seller_idx]

				# Check if buyer is willing to pay more or equal to seller's price
				if buyer_price >= seller_price:
					# Determine the quantity to be matched
					matched_quantity = min(sorted_q_u[buyer_idx], sorted_q_g[seller_idx])

					# Reduce quantities of both buyer and seller
					sorted_q_u[buyer_idx] -= matched_quantity
					sorted_q_g[seller_idx] -= matched_quantity

					# Record the seller's price for the matched quantity
					final_prices_sellers[sorted_sell_indices[seller_idx]] = seller_price

					# Move to the next buyer if their quantity is fully matched
					if sorted_q_u[buyer_idx] == 0:
						buyer_idx += 1

					# Move to the next seller if their quantity is fully matched
					if sorted_q_g[seller_idx] == 0:
						seller_idx += 1
				else:
					# If no match is possible, stop the auction
					break

			# Final accepted quantities (already adjusted in sorted_q_u and sorted_q_g)
			q_accepted_u = np.zeros_like(q_u)
			q_accepted_u[sorted_buy_indices] = q_u[sorted_buy_indices] - sorted_q_u

			q_accepted_g = np.zeros_like(q_g)
			q_accepted_g[sorted_sell_indices] = q_g[sorted_sell_indices] - sorted_q_g

		else:
			# Case where total supply is insufficient to meet demand
			LL = True  # Load loss
			
			final_prices_sellers = np.full_like(price_g, price_cap)  # Sellers get the capped price
			final_price = price_cap  # Set price to cap

			# Accept all seller quantities
			q_accepted_g = q_g_o

			# Adjust buyer quantities proportionally to available supply
			q_accepted_u = q_u * total_supply / (total_demand + self.delta_small)
			
		return final_prices_sellers, q_accepted_u, q_accepted_g, LL

	""" General action-masking function for generator actions """
		
	def get_action_masks_g(self, investments_enabled, capacity_existing_storage_lt):
		

		possible_actions = np.ones(int(self.step_g_inv * self.act_inv + self.step_g_bids * self.act_bids_cm + self.step_g_inv * self.act_inv_cm
											  + self.step_g_bids * self.act_bids_CfD + self.step_g_inv * self.act_inv_CfD + self.step_SoC_control * self.act_SoC))


		invalid_actions = []

		# Investment not enabled out of period

		for j in range(self.act_inv):

			if j < self.n_tech:

				construction_time = self.construction_time[j]
			
			else:

				construction_time = self.construction_time_battery

			if self.month != 0 or self.max_year - self.year - 5 <=  construction_time or investments_enabled[j] == 0: 

				# Merchant investment 
			
				invalid_actions_temp = list(range(1 + self.step_g_inv * (j), 
									  self.step_g_inv * (j + 1)))
				
				invalid_actions = invalid_actions + invalid_actions_temp
		
			else:
				
				# Merchant investment 
			
				invalid_actions_temp = list(range(self.step_g_inv * (j) + 0, 
									  self.step_g_inv * (j) + 1))
				
				invalid_actions = invalid_actions + invalid_actions_temp

		##  Capacity market masking 

		for j in range(self.act_bids_cm):

			if j < self.n_tech:

				construction_time = self.construction_time[j]
			
			else:

				construction_time = self.construction_time_battery
		
			if self.month != 2 or self.cm_auction_indicator == False or self.cm_tech_cc[0,j] <= 0.02 or self.max_year - self.year - 5 <= construction_time or investments_enabled[j] == 0 or self.CM_activation == False:

				# Capacity Market prices 
				
				invalid_actions_temp = list(range(1 + self.step_g_inv * self.act_inv + self.step_g_bids * (j), 
									  self.step_g_inv * self.act_inv + self.step_g_bids * (j + 1)))
					
				invalid_actions = invalid_actions + invalid_actions_temp

				# Capacity Market quantities 
				
				invalid_actions_temp = list(range(1 + self.step_g_inv * self.act_inv + self.step_g_bids * self.act_bids_cm + self.step_g_inv * (j),
									  self.step_g_inv * self.act_inv + self.step_g_bids * self.act_bids_cm + self.step_g_inv * (j + 1)))
					
				invalid_actions = invalid_actions + invalid_actions_temp
			
			else:

				# Capacity Market prices 
				
				invalid_actions_temp = list(range(self.step_g_inv * self.act_inv + self.step_g_bids * (j) + 0, 
												+ self.step_g_inv * self.act_inv + self.step_g_bids * (j) + 1))
					
				invalid_actions = invalid_actions + invalid_actions_temp

				# Capacity Market quantities 
				
				invalid_actions_temp = list(range(self.step_g_inv * self.act_inv + self.step_g_bids * self.act_bids_cm + self.step_g_inv * (j) + 0,
									  self.step_g_inv * self.act_inv + self.step_g_bids * self.act_bids_cm + self.step_g_inv * (j) + 1))
					
				invalid_actions = invalid_actions + invalid_actions_temp

		##  CfD masking

		for j in range(self.act_bids_CfD):
		
			if self.month != 4 or self.CfD_auction_indicator == False or self.max_year - self.year - 5 <= self.construction_time[j] or investments_enabled[j] == 0 or self.CfD_activation == False:

				# CfD Market prices 
				
				invalid_actions_temp = list(range(1 + self.step_g_inv * self.act_inv + self.step_g_bids * self.act_bids_cm + self.step_g_inv * self.act_inv_cm
												+ self.step_g_bids * (j),
										self.step_g_inv * self.act_inv + self.step_g_bids * self.act_bids_cm + self.step_g_inv * self.act_inv_cm
												+ self.step_g_bids * (j + 1),))
					
				invalid_actions = invalid_actions + invalid_actions_temp

				# CfD Market quantities 
				
				invalid_actions_temp = list(range(1 + self.step_g_inv * self.act_inv + self.step_g_bids * self.act_bids_cm + self.step_g_inv * self.act_inv_cm
												+ self.step_g_bids * self.act_bids_CfD + self.step_g_inv * (j), 
										self.step_g_inv * self.act_inv + self.step_g_bids * self.act_bids_cm + self.step_g_inv * self.act_inv_cm
												+ self.step_g_bids * self.act_bids_CfD + self.step_g_inv * (j + 1)))
					
				invalid_actions = invalid_actions + invalid_actions_temp
			
			else:

				# CfD Market prices
				
				invalid_actions_temp = list(range(self.step_g_inv * self.act_inv + self.step_g_bids * self.act_bids_cm + self.step_g_inv * self.act_inv_cm 
												+ self.step_g_bids * (j) + 0,
										self.step_g_inv * self.act_inv + self.step_g_bids * self.act_bids_cm + self.step_g_inv * self.act_inv_cm
												+ self.step_g_bids * (j) + 1,))
					
				invalid_actions = invalid_actions + invalid_actions_temp

				# CfD Market quantities 
				
				invalid_actions_temp = list(range(self.step_g_inv * self.act_inv + self.step_g_bids * self.act_bids_cm + self.step_g_inv * self.act_inv_cm
												+ self.step_g_bids * self.act_bids_CfD + self.step_g_inv * (j) + 0, 
										self.step_g_inv * self.act_inv + self.step_g_bids * self.act_bids_cm + self.step_g_inv * self.act_inv_cm
												+ self.step_g_bids * self.act_bids_CfD + self.step_g_inv * (j) + 1))
				
				invalid_actions = invalid_actions + invalid_actions_temp

		if capacity_existing_storage_lt <= 0.2:
			
			invalid_actions_temp = list(range(1 + self.step_g_inv * self.act_inv + self.step_g_bids * self.act_bids_cm + self.step_g_inv * self.act_inv_cm
												+ self.step_g_bids * self.act_bids_CfD + self.step_g_inv * self.act_inv_CfD,
										self.step_g_inv * self.act_inv + self.step_g_bids * self.act_bids_cm + self.step_g_inv * self.act_inv_cm
												+ self.step_g_bids * self.act_bids_CfD + self.step_g_inv * self.act_inv_CfD + self.act_SoC * self.step_SoC_control))
			
			invalid_actions = invalid_actions + invalid_actions_temp

		## Aggregating invalid actions

		for i in invalid_actions:   

			possible_actions[i] = 0
			
		return np.float16(possible_actions)
	

	""" CfD balance estimation"""

	def CfD_balance_estimation(self):
		
		total_RES_energy = 0

		CfD_fc_aging = 0

		for j in range(self.n_tech_RES):

			# Summing existing capacity
			
			total_RES_energy += np.sum(self.inv_g[:,j]) * self.availability_tech_average_yearly[self.year_int + self.planning_horizon, j] * self.average_failures[0,j]

			# Suming capacity under construction

			total_RES_energy += (self.cm_fc_under_construction[j]) * self.availability_tech_average_yearly[self.year_int + self.planning_horizon, j]  * self.average_failures[0,j]

			# Discounting existing capacity

			inv_sum = np.sum(self.inv_init_g[:, j])
			
			for year_temp in range(int(self.planning_horizon)):

				CfD_fc_aging += inv_sum * self.aging[self.year_int + year_temp, j] * self.availability_tech_average_yearly[self.year_int + self.planning_horizon, j]  * self.average_failures[0,j]

		total_RES_energy -= CfD_fc_aging

		# Sum of average hydro generation

		total_RES_energy += self.availability_tech_average_yearly[self.year_int + self.planning_horizon, j]  * np.sum(self.capacity_existing_storage_lt) * self.average_failures[0,self.n_tech] 

		# Adding run of the river hydro

		total_RES_energy += self.availability_tech_average_yearly[self.year_int + self.planning_horizon, self.n_tech + 1] * self.average_failures[0,self.n_tech]  

		# Demand 

		future_res_target_demand = self.demand_average_year[self.year_int + self.planning_horizon] * self.CfD_target[self.year_int + self.planning_horizon]/100

		# Balance calculation

		CfD_balance = (future_res_target_demand - total_RES_energy)

		CfD_balance_real = (future_res_target_demand - total_RES_energy)

		if CfD_balance > 0:
			CfD_auction_indicator = True
		else:
			CfD_auction_indicator = False

		return CfD_balance, CfD_auction_indicator, CfD_balance_real
	
	""" Aggregated failures for snort-term asset availability """
		
	def calculate_aggregated_failures(self):

		agent_multiple = np.ceil(self.inv_g / 250).astype(int)

		aggregated_failures = np.zeros([self.agents_g, self.n_tech])

		for i in range(self.agents_g):

			for j in range(self.n_tech):

				failure_temp = 0

				for t in range(agent_multiple[i,j]):

					s_failures = self.random_number_failures[self.time_random_failures]

					failure_temp += (1 + self.scenario_failures[j, s_failures]/100)/agent_multiple[i,j] 

					self.time_random_failures += 1

					if self.time_random_failures >= 52100:
						self.time_random_failures = 0

				aggregated_failures[i,j] = failure_temp

		return aggregated_failures
	
	def top_n_rows(self, matrix, n, high):

		# Sort elements by value (ascending for lowest, descending for highest)
		sorted_indices = np.argsort(matrix, axis=None)
		if high:
			sorted_indices = sorted_indices[::-1]  # Reverse for descending order
		
		# Convert flattened indices to row indices without repetition
		unique_rows = set()
		row_indexes = []
		for idx in sorted_indices:
			row, col = divmod(idx, matrix.shape[1])
			if row not in unique_rows:
				unique_rows.add(row)
				row_indexes.append(row)
			if len(row_indexes) == n:  # Stop after collecting `n` unique rows
				break
		
		return row_indexes
	
	""" Capacity market balance estimation """
	
	def cm_balance_estimation(self):

		# Adequacy calculations are carried out in conditions with maximum demand, and resource availability under those conditions

		scenario_cm = 4

		max_index = int(self.short_t * self.yearly_resolution)

		availability_tech_cm = np.zeros([max_index, self.n_tech + 1])

		total_availability = np.zeros([max_index, self.n_tech])

		demand_cm = np.zeros([max_index]) 

		# Availability and demand per scenario
		
		for month_temp in range(self.yearly_resolution):

			index_init = (self.year_int  + self.planning_horizon) * self.short_t * self.yearly_resolution * 5 + month_temp * self.short_t * 5 + scenario_cm * self.short_t

			index_final = (self.year_int  + self.planning_horizon) * self.short_t * self.yearly_resolution * 5 + month_temp * self.short_t * 5 + (scenario_cm + 1) * self.short_t

			availability_tech_cm[self.short_t * month_temp:self.short_t * (month_temp + 1),:] = self.availability_tech[int(index_init):int(index_final),:self.n_tech + 1]
	
			demand_cm[self.short_t * (month_temp + 0):self.short_t * (month_temp + 1)] = self.demand[int(index_init):int(index_final)] - self.availability_tech[int(index_init):int(index_final),self.n_tech + 1]

		for j in range(self.n_tech):
			
			total_availability[:,j] = (availability_tech_cm[:, j] * self.average_failures[0,j]) * (np.sum(self.inv_g[:, j]) + self.cm_fc_under_construction[j])

		# Discount of aging existing capacity for the planning period

		cm_fc_aging = np.zeros([max_index, self.n_tech])

		for j in range(self.n_tech):

			inv_sum = np.sum(self.inv_init_g[:, j])
			
			for year_temp in range(self.planning_horizon):

				cm_fc_aging[:,j] += inv_sum * self.aging[int(self.year_int + year_temp), j] * (availability_tech_cm[:, j]) * self.average_failures[0,j]

		total_availability -= cm_fc_aging

		margin_factor = (1 + self.cm_excess_demand)

		hourly_balance = self.storage_profiles_cm(total_availability, margin_factor, demand_cm, availability_tech_cm)

		cm_balance = np.max(hourly_balance)

		cm_balance_real = np.max(hourly_balance)

		index_max = np.argmax(hourly_balance)
	
		# Regulator checks demand projections and compares it with assets under construction, and existing capacity (with demand 4 years in advance)
		cm_auction_indicator = False 

		cm_tech_cc = np.zeros([1, self.n_tech + 1])

		cm_tech_cc[0,:self.n_tech] = availability_tech_cm[index_max, :self.n_tech] * self.average_failures[0,:self.n_tech]

		cm_tech_cc[0,self.n_tech] = 0.95

		if cm_balance > 0:
			cm_auction_indicator = True
		else:
			cm_auction_indicator = False

		return cm_balance, cm_auction_indicator, cm_tech_cc, hourly_balance, cm_balance_real
	
	""" Calculating storage profiles """

	def storage_profiles_cm(self, total_availability, margin_factor, demand_cm, availability_tech_cm):

		max_index = int(self.short_t * self.yearly_resolution)

		hourly_balance = np.zeros([max_index])

		self.strategy_charge_3_hours = np.zeros([self.short_t])
		self.strategy_discharge_3_hours = np.zeros([self.short_t])

		self.strategy_charge_4_hours = np.zeros([self.short_t])
		self.strategy_discharge_4_hours = np.zeros([self.short_t])

		self.strategy_charge_8_hours = np.zeros([self.short_t])
		self.strategy_discharge_8_hours = np.zeros([self.short_t])

		iters_max_battery = 10

		average_hydro = self.availability_tech_average_yearly[self.year_int, self.n_tech]

		for iters in range(iters_max_battery):

			demand_copy = copy.deepcopy(demand_cm)	

			for t in range (max_index):

				demand_temp = demand_copy[t]

				## Priority to short_term storage

				demand_temp += (self.strategy_charge_3_hours[np.mod(t,self.short_t-1)] - self.strategy_discharge_3_hours[np.mod(t,self.short_t-1)]) * (np.sum(self.capacity_cm_battery) + self.cm_fc_under_construction_battery_cm) * self.average_failures[0,self.n_tech]

				demand_temp += (self.strategy_charge_4_hours[np.mod(t,self.short_t-1)] - self.strategy_discharge_4_hours[np.mod(t,self.short_t-1)]) * (np.sum(self.capacity_merchant_battery) + self.cm_fc_under_construction_battery_merchant) * self.average_failures[0,self.n_tech]

				demand_temp += (self.strategy_charge_8_hours[np.mod(t,self.short_t-1)] - self.strategy_discharge_8_hours[np.mod(t,self.short_t-1)] - availability_tech_cm[t, self.n_tech]) * np.sum(self.capacity_existing_storage_lt) * self.average_failures[0,self.n_tech]

				hourly_balance[t] = demand_temp * margin_factor - np.sum(total_availability[t,:])

			hourly_balance_matrix = np.transpose(hourly_balance.reshape((self.yearly_resolution, self.short_t)))

			top_indexes_3_hours = self.top_n_rows(hourly_balance_matrix, 3, True)

			low_indexes_3_hours = self.top_n_rows(hourly_balance_matrix, 3, False)

			self.strategy_discharge_3_hours[top_indexes_3_hours] += 1/iters_max_battery

			self.strategy_charge_3_hours[low_indexes_3_hours] += 1/iters_max_battery

			top_indexes_4_hours = self.top_n_rows(hourly_balance_matrix, 3, True)

			low_indexes_4_hours = self.top_n_rows(hourly_balance_matrix, 3, False)

			self.strategy_discharge_4_hours[top_indexes_4_hours] += 1/iters_max_battery

			self.strategy_charge_4_hours[low_indexes_4_hours] += 1/iters_max_battery

			top_indexes_8_hours = self.top_n_rows(hourly_balance_matrix, 8, True)

			low_indexes_8_hours = self.top_n_rows(hourly_balance_matrix, 8, False)

			self.strategy_discharge_8_hours[top_indexes_8_hours] += 1/iters_max_battery - average_hydro/iters_max_battery

			self.strategy_charge_8_hours[low_indexes_8_hours] += 1/iters_max_battery - average_hydro/iters_max_battery

		demand_copy = copy.deepcopy(demand_cm)	

		for t in range (max_index):

			demand_temp = demand_copy[t]

			## Priority to short_term storage

			demand_temp += (self.strategy_charge_3_hours[np.mod(t,self.short_t-1)] - self.strategy_discharge_3_hours[np.mod(t,self.short_t-1)]) * (np.sum(self.capacity_cm_battery) + self.cm_fc_under_construction_battery_cm) * self.average_failures[0,self.n_tech]

			demand_temp += (self.strategy_charge_4_hours[np.mod(t,self.short_t-1)] - self.strategy_discharge_4_hours[np.mod(t,self.short_t-1)]) * (np.sum(self.capacity_merchant_battery)+ self.cm_fc_under_construction_battery_merchant) * self.average_failures[0,self.n_tech]

			# Hydro adjustment 

			demand_temp += (self.strategy_charge_8_hours[np.mod(t,self.short_t-1)] - self.strategy_discharge_8_hours[np.mod(t,self.short_t-1)] - availability_tech_cm[t, self.n_tech]) * np.sum(self.capacity_existing_storage_lt) * self.average_failures[0,self.n_tech]

			hourly_balance[t] = demand_temp * margin_factor - np.sum(total_availability[t,:])

		return hourly_balance
	
	""" Render method - for compatibility """
	
	def render(self):
		pass
