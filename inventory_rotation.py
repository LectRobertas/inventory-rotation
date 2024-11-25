# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 13:26:08 2024

@author: Win-10
"""

# In[0]:

import glob
import pandas as pd
import numpy as np
import random

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
#from stable_baselines3.common.vec_env import DummyVecEnv
    
"""
    0. DEVICE CHECK & CONFIG
"""
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

import torch
print("PyTorch is using GPU: ", torch.cuda.is_available())
print("GPU Name: ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")


# In[1]:

"""
    1. DATA SELECTION
"""
# Path to the data (folders)
path = ['C:/Users/Win-10/Documents/Python Scripts/ku']

frames = []
for p in path:
    all_files = glob.glob(p + "/history_*.csv")
    for filename in all_files:
        df = pd.read_csv(filename, header=0, sep=',', index_col=None, low_memory=False)
        frames.append(df)

df_history = pd.concat(frames)

df_items = pd.read_csv(p + "/skus.csv", header=0, sep=',', index_col=None, low_memory=False)
df_locations = pd.read_csv(p + "/locations.csv", header=0, sep=',', index_col=None, low_memory=False)

# Remove temp variables from workspace
del path, all_files, frames, filename, p, df
    
# Step 1: Filter df_items by multiple skuName values
sku_names_to_filter = ['Kūno apsauginis kremas'] # ['Lūpų dažai','Parfumuotas vanduo (EDP)','Kreminė pudra','Dieninis kremas','Lūpų blizgesiai','Akių šešėliai mono 1sp.','Tualetinis vanduo (EDT)']  # Example skuNames to filter by
df_items = df_items[df_items['categoryGroup'].isin(sku_names_to_filter)]

# Step 2: Filter df_history by the skuid values of the filtered df_items
df_history = df_history[df_history['skuID'].isin(df_items['skuid'])]

# Set in chronological order
df_history['updateDate'] = pd.to_datetime(df_history['updateDate'])
df_history = df_history.sort_values(by='updateDate').reset_index(drop=True)


# In[2]:

"""
    2. PREPARE CUSTOM RL ENVIRONMENT
"""

class InventoryRotationEnv(gym.Env):
    def __init__(self, df_items, df_history):
        super(InventoryRotationEnv, self).__init__()
        self.df_items = df_items
        self.df_history = df_history
        self.num_skus = len(df_items)
        
        # Define action space: choose an SKU to add or remove
        self.num_skus = len(df_items)
        self.action_space = spaces.Discrete(self.num_skus * 2)  # num_skus for remove, num_skus for add
        
        # Calculate maximum day value dynamically based on the dataset
        max_day_value = (df_history['updateDate'].max() - df_history['updateDate'].min()).days + 1
        
        # Define observation space: SKU ID and day of the year
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0]),  # SKU ID and day of the year
            high=np.array([self.num_skus - 1, max_day_value, 1, np.inf, np.inf]),
            dtype=np.float32
        )
        
        # Initialize state
        self.state = None
        self.current_step = 0
        self.max_steps = len(df_history['updateDate'].unique())  # Set max steps to cover the length of history
        self.current_inventory = set(random.sample(list(df_items['skuid']), 10))  # Start with a random set of 10 SKUs

    def reset(self, *, seed=None):
        # Reset environment to initial state
        self.current_step = 0
        
        self.current_inventory = set(random.sample(list(self.df_items['skuid']), 10))
        self.current_step_date = pd.to_datetime(self.df_history['updateDate'].iloc[self.current_step])
        self.state = self._next_observation()
        return self.state, {}
        
    def _next_observation(self):
        # Filter history for the current day
        day_history = self.df_history[self.df_history['updateDate'] == self.current_step_date]
    
        # Select a SKU from the SKUs available on the current day
        if not day_history.empty:
            # Pick a random SKU from the SKUs that are recorded in history for the current day
            sku_id = random.choice(day_history['skuID'].unique())
        else:
            # If no SKU data for the day, default to a random SKU
            sku_id = random.choice(self.df_items['skuid'])
    
        # Day of the year from the current date
        day_of_year = self.current_step_date.timetuple().tm_yday
    
        # Check if the SKU was sold yesterday
        if self.current_step > 0:
            # Get the previous day's history
            previous_day = self.current_step_date - pd.Timedelta(days=1)
            previous_day_history = self.df_history[self.df_history['updateDate'] == previous_day]
            was_sold = 1 if not previous_day_history[previous_day_history['skuID'] == sku_id]['consumption'].empty else 0
        else:
            was_sold = 0
    
        # Current stock level for the SKU (if available)
        current_stock = day_history[day_history['skuID'] == sku_id]['atSiteQnt'].sum() if not day_history.empty else 0
    
        # Recent sales (e.g., total sales in the past 7 days)
        recent_sales = 0
        if self.current_step > 7:
            past_week = self.df_history[
                (self.df_history['updateDate'] >= self.current_step_date - pd.Timedelta(days=7)) &
                (self.df_history['updateDate'] < self.current_step_date) &
                (self.df_history['skuID'] == sku_id)
            ]
            recent_sales = past_week['consumption'].sum()
    
        # Return the observation, including SKU ID, day of the year, was_sold indicator, stock level, and recent sales
        return np.array([sku_id, day_of_year, was_sold, current_stock, recent_sales], dtype=np.float32)
    
    def step(self, action):
        # Determine if action is adding or removing SKU
        if action < self.num_skus:
            sku_to_remove = self.df_items.iloc[action]['skuid']
            if sku_to_remove in self.current_inventory:
                self.current_inventory.remove(sku_to_remove)
        else:
            sku_to_add = self.df_items.iloc[action - self.num_skus]['skuid']
            day_history = self.df_history[self.df_history['updateDate'] == self.current_step_date]
            sku_quantity = day_history[day_history['skuID'] == sku_to_add]['atSiteQnt'].sum()
            if sku_quantity > 0:
                self.current_inventory.add(sku_to_add)
    
        # Calculate reward based on profit
        reward = self._calculate_profit()
    
        # Update state
        self.current_step += 1
        if self.current_step < len(self.df_history):
            self.current_step_date = self.df_history['updateDate'].iloc[self.current_step]
        done = self.current_step >= self.max_steps
        self.state = self._next_observation()
    
        # Extract the current stock level and recent sales from the updated state
        sku_id, day_of_year, was_sold, current_stock, recent_sales = self.state
    
        return self.state, reward, done, False, {
            'current_inventory': list(self.current_inventory), 
            'current_step_date': self.current_step_date,
            'sku_id': sku_id,
            'day_of_year': day_of_year,
            'was_sold': was_sold,
            'current_stock': current_stock,
            'recent_sales': recent_sales
        }
    
    def _calculate_profit(self):
        # Define the window for calculating profit
        window_size = 7
        profit = 0
    
        # Calculate the start date for the rolling window
        start_date = self.current_step_date - pd.Timedelta(days=window_size - 1)
        
        # Filter history to only include data within the rolling window ending on the current day
        rolling_window_history = self.df_history[
            (self.df_history['updateDate'] >= start_date) &
            (self.df_history['updateDate'] <= self.current_step_date)
        ]
    
        # Calculate profit for all SKUs in the current inventory over the rolling window
        for skuid in self.current_inventory:
            sku_sales_events = rolling_window_history[rolling_window_history['skuID'] == skuid]
            if sku_sales_events.empty:
                continue
    
            # Calculate profit for each sales event
            for _, sale in sku_sales_events.iterrows():
                sku_sales = sale['consumption']
                sku_price = sale['SalesPrice'] - sale['PurchasingPrice']  # Considering profit margin
                profit += sku_sales * sku_price
    
        print(f"Calculated profit for rolling window ending on {self.current_step_date}: {profit}")
        return profit


# In[3]:

"""
    3. CREATE AND TRAIN RL ENV
"""

env = InventoryRotationEnv(df_items, df_history)

model = DQN('MlpPolicy', env, verbose=1) # With default hyperparameters
"""model = DQN(  
    'MlpPolicy',
    env,
    learning_rate=0.0005,
    buffer_size=50000,
    batch_size=64,
    gamma=0.99,
    exploration_fraction=0.1,
    exploration_final_eps=0.02,
    target_update_interval=1000,
    verbose=1
)"""
model.learn(total_timesteps=10000)  # Is the total number of samples (env steps) to train on.

# Save/Load the trained model
model.save("inventory_rotation_dqn")
#model = DQN.load("inventory_rotation_dqn")  

# In[4]:

"""
    4. TEST CASE WITH A SPECIFIC SKU
"""

def set_environment_to_date(env, specific_date):
    date_index = env.df_history[env.df_history['updateDate'] == specific_date].index[0]
    env.current_step = date_index
    env.current_step_date = pd.to_datetime(env.df_history['updateDate'].iloc[env.current_step])
    env.state = env._next_observation()
    return env.state

specific_date = '2023-01-01'
obs = set_environment_to_date(env, specific_date)

# Set the observation for the specific SKU ID 808
sku_id_to_check = 44333
day_of_year = pd.to_datetime(specific_date).timetuple().tm_yday
obs = np.array([sku_id_to_check, day_of_year], dtype=np.float32)

action, _states = model.predict(obs, deterministic=True)
obs, reward, done, term, info = env.step(action)

if action < env.num_skus:
    print(f"On {specific_date}, the model suggests removing SKU ID {sku_id_to_check}.")
else:
    print(f"On {specific_date}, the model suggests adding SKU ID {sku_id_to_check}.")
