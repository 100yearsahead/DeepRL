"""
Simple discrete microgrid environment.

Idea:
We are modelling a tiny solar microgrid with a battery.

At each step, the agent sees:
- how full the battery is
- whether demand is low/medium/high
- whether solar is low/medium/high
- what part of the day it is

The agent can:
- charge the battery
- do nothing
- discharge the battery

The goal is to learn a sensible battery policy:
charge when solar is useful, discharge when demand is high,
and avoid relying too much on the grid.
"""

import numpy as np


class DiscreteMicrogridEnv:
    def __init__(self, battery_capacity=10, max_grid_import=2.0, episode_length=24, seed=None):
        self.battery_capacity = battery_capacity
        self.episode_length = episode_length

        # The grid can only cover up to this amount of demand per step.
        # This makes unmet demand possible if demand is high and the battery is empty.
        self.max_grid_import = max_grid_import
        self.rng = np.random.default_rng(seed)


        # Actions the RL agent can choose.
        self.actions = {
            0: "charge",
            1: "idle",
            2: "discharge",
        }

        # We keep demand and solar simple for tabular Q-learning.
        # These numbers are not exact real-world values.
        # They are simple regime values: low, medium, high.
        self.demand_values = {
            0: 1.0,  # low demand
            1: 2.0,  # medium demand
            2: 3.0,  # high demand
        }

        self.solar_values = {
            0: 0.0,  # low / no solar
            1: 1.5,  # medium solar
            2: 3.0,  # high solar
        }

        self.time_labels = {
            0: "night",
            1: "morning",
            2: "midday",
            3: "evening",
        }

        self.demand_labels = {
            0: "low",
            1: "medium",
            2: "high",
        }

        self.solar_labels = {
            0: "low",
            1: "medium",
            2: "high",
        }

        self.reset()

    def reset(self):
        """
        Start a new episode.

        We begin at hour 0 with the battery half full.
        """
        self.hour = 0
        self.steps_taken = 0
        self.battery_level = self.battery_capacity // 2

        self.time_bin = self.get_time_bin(self.hour)
        self.demand_regime = self.sample_demand(self.time_bin)
        self.solar_regime = self.sample_solar(self.time_bin)

        return self.get_state()

    def step(self, action):
        """
        Take one action in the environment.

        Returns:
            next_state, reward, done, info
        """
        action = int(action)  #  Convert NumPy integers into normal Python integers.
        
        if action not in self.actions:
            raise ValueError("Action must be 0=charge, 1=idle, or 2=discharge.")

        

        demand = self.demand_values[self.demand_regime]
        solar = self.solar_values[self.solar_regime]

        # These are tracked so we can analyse behaviour later.
        battery_used = 0.0
        grid_import = 0.0
        wasted_solar = 0.0

        # -----------------------------------
        # Action 0: charge battery
        # -----------------------------------
        if action == 0:
            # Only charge from surplus solar.
            surplus_solar = max(0.0, solar - demand)

            if surplus_solar > 0 and self.battery_level < self.battery_capacity:
                self.battery_level += 1
                charge_amount = 1.0
            else:
                charge_amount = 0.0

            # If solar cannot meet demand, grid covers the shortfall.
            grid_import = self.get_grid_import(demand - solar)

            # Any surplus solar not stored is wasted.
            wasted_solar = max(0.0, surplus_solar - charge_amount)

        # -----------------------------------
        # Action 1: idle
        # -----------------------------------
        elif action == 1:
            # Battery does nothing.
            # Grid covers any shortage.
            grid_import = self.get_grid_import(demand - solar)

            # Extra solar is wasted if not stored.
            wasted_solar = max(0.0, solar - demand)

        # -----------------------------------
        # Action 2: discharge battery
        # -----------------------------------
        elif action == 2:
            if self.battery_level > 0:
                self.battery_level -= 1
                battery_used = 1.0

            # After solar and battery, grid covers the remaining demand.
            grid_import =   self.get_grid_import(demand - solar - battery_used)

            # If solar already exceeds demand, extra solar is wasted.
            wasted_solar = max(0.0, solar - demand)

        reward, reward_parts = self.calculate_reward(
            demand=demand,
            solar=solar,
            battery_used=battery_used,
            grid_import=grid_import,
            wasted_solar=wasted_solar,
        )

        # Move time forward.
        self.hour += 1
        self.steps_taken += 1

        done = self.steps_taken >= self.episode_length

        # Sample the next demand and solar conditions.
        self.time_bin = self.get_time_bin(self.hour)
        self.demand_regime = self.sample_demand(self.time_bin)
        self.solar_regime = self.sample_solar(self.time_bin)

        next_state = self.get_state()

        info = {
            "hour": self.hour,
            "action": action,
            "action_name": self.actions[action],
            "demand": demand,
            "solar": solar,
            "battery_level": self.battery_level,
            "battery_used": battery_used,
            "grid_import": grid_import,
            "wasted_solar": wasted_solar,
        }

        info.update(reward_parts)

        return next_state, reward, done, info

    def get_state(self):
        """
        State is a tuple, so it can be used directly in a Q-table.

        Format:
        (battery_level, demand_regime, solar_regime, time_bin)
        """
        return (
            self.battery_level,
            self.demand_regime,
            self.solar_regime,
            self.time_bin,
        )
    def get_grid_import(self, remaining_demand):
        """
        Grid import is limited.

        If the remaining demand is larger than this limit, the rest becomes unmet demand.
        This makes the environment more like a real load-balancing problem.
        """
        return min(self.max_grid_import, max(0.0, remaining_demand))

    def get_time_bin(self, hour):
        """
        Convert hour into a rough time-of-day category.

        0 = night
        1 = morning
        2 = midday
        3 = evening
        """
        hour = hour % 24

        if hour < 6:
            return 0  # night
        elif hour < 12:
            return 1  # morning
        elif hour < 18:
            return 2  # midday
        else:
            return 3  # evening

    def sample_demand(self, time_bin):
        """
        Demand is more likely to be high in the morning and evening.
        This is a simplified version of real daily demand patterns.
        """
        demand_probs = {
            0: [0.6, 0.3, 0.1],  # night: mostly low
            1: [0.2, 0.5, 0.3],  # morning: medium/high
            2: [0.3, 0.5, 0.2],  # midday: mostly medium
            3: [0.1, 0.4, 0.5],  # evening: often high
        }

        return int(self.rng.choice([0, 1, 2], p=demand_probs[time_bin]))

    def sample_solar(self, time_bin):
        """
        Solar is strongest around midday and absent/low at night.
        Again, this is simplified but realistic enough for this task.
        """
        solar_probs = {
            0: [1.0, 0.0, 0.0],  # night: no solar
            1: [0.3, 0.6, 0.1],  # morning: mostly medium
            2: [0.1, 0.3, 0.6],  # midday: often high
            3: [0.7, 0.3, 0.0],  # evening: mostly low
        }

        return int(self.rng.choice([0, 1, 2], p=solar_probs[time_bin]))

    def calculate_reward(self, demand, solar, battery_used, grid_import, wasted_solar):
        """
        Reward function.

        We punish:
        - unmet demand: very bad
        - grid import: costs money / less local self-sufficiency
        - wasted solar: inefficient
        - battery use: small wear cost

        Grid import is limited, so unmet demand can happen when demand is high,
        solar is low, and the battery cannot cover the shortage.
        """
        unmet_demand = max(0.0, demand - solar - battery_used - grid_import)

        unmet_penalty = -10.0 * unmet_demand
        grid_penalty = -1.0 * grid_import
        solar_waste_penalty = -0.5 * wasted_solar
        battery_penalty = -0.1 * abs(battery_used)

        reward = (
            unmet_penalty
            + grid_penalty
            + solar_waste_penalty
            + battery_penalty
        )

        reward_parts = {
            "unmet_demand": unmet_demand,
            "unmet_penalty": unmet_penalty,
            "grid_penalty": grid_penalty,
            "solar_waste_penalty": solar_waste_penalty,
            "battery_penalty": battery_penalty,
        }

        return reward, reward_parts

    def describe_state(self, state=None):
        """
        Human-readable version of a state.
        Useful for debugging and report examples.
        """
        if state is None:
            state = self.get_state()

        battery, demand, solar, time_bin = state

        return {
            "battery_level": battery,
            "demand": self.demand_labels[demand],
            "solar": self.solar_labels[solar],
            "time": self.time_labels[time_bin],
        }

    def state_space_shape(self):
        """
        Number of possible values for each state component.
        """
        return (
            self.battery_capacity + 1,
            3,  # demand regimes
            3,  # solar regimes
            4,  # time bins
        )

    def num_actions(self):
        return len(self.actions)