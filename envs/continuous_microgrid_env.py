"""
Continuous microgrid environment for DQN.

This is the advanced version of the basic discrete environment.

Basic environment:
- demand was low / medium / high
- solar was low / medium / high
- battery was 0, 1, ..., 10

Advanced environment:
- demand is a continuous value
- solar is a continuous value
- battery is a continuous state of charge between 0 and 1

The action space is still discrete:
0 = charge
1 = idle
2 = discharge

This keeps the problem suitable for DQN.
"""

import numpy as np


class ContinuousMicrogridEnv:
    def __init__(
        self,
        episode_length=24,
        battery_capacity=10.0,
        max_grid_import=2.0,
        max_charge_rate=1.0,
        max_discharge_rate=1.0,
        demand_noise=0.15,
        solar_noise=0.15,
        solar_scale=1.0,
        seed=None,
    ):
        self.episode_length = episode_length
        self.battery_capacity = battery_capacity
        self.max_grid_import = max_grid_import
        self.max_charge_rate = max_charge_rate
        self.max_discharge_rate = max_discharge_rate

        # These let us make stress-test environments later.
        self.demand_noise = demand_noise
        self.solar_noise = solar_noise
        self.solar_scale = solar_scale

        self.rng = np.random.default_rng(seed)

        self.actions = {
            0: "charge",
            1: "idle",
            2: "discharge",
        }

        self.reset()

    def reset(self):
        """
        Start a new 24-hour episode.
        """
        self.hour = 0
        self.steps_taken = 0

        # Battery starts half full.
        self.battery_energy = 0.5 * self.battery_capacity

        self.demand = self.sample_demand(self.hour)
        self.solar = self.sample_solar(self.hour)
        self.grid_price = self.get_grid_price(self.hour)

        return self.get_state()

    def step(self, action):
        """
        Take one action in the environment.

        Returns:
            next_state, reward, done, info
        """
        action = int(action)

        if action not in self.actions:
            raise ValueError("Action must be 0=charge, 1=idle, or 2=discharge.")

        demand = self.demand
        solar = self.solar
        price = self.grid_price

        battery_used = 0.0
        grid_import = 0.0
        wasted_solar = 0.0

        # -------------------------
        # Action 0: charge battery
        # -------------------------
        if action == 0:
            surplus_solar = max(0.0, solar - demand)

            free_space = self.battery_capacity - self.battery_energy
            charge_amount = min(self.max_charge_rate, surplus_solar, free_space)

            self.battery_energy += charge_amount

            grid_import = self.get_grid_import(demand - solar)
            wasted_solar = max(0.0, surplus_solar - charge_amount)

        # -------------------------
        # Action 1: idle
        # -------------------------
        elif action == 1:
            grid_import = self.get_grid_import(demand - solar)
            wasted_solar = max(0.0, solar - demand)

        # -------------------------
        # Action 2: discharge battery
        # -------------------------
        elif action == 2:
            needed_after_solar = max(0.0, demand - solar)

            discharge_amount = min(
                self.max_discharge_rate,
                self.battery_energy,
                needed_after_solar,
            )

            self.battery_energy -= discharge_amount
            battery_used = discharge_amount

            grid_import = self.get_grid_import(demand - solar - battery_used)
            wasted_solar = max(0.0, solar - demand)

        reward, reward_parts = self.calculate_reward(
            demand=demand,
            solar=solar,
            battery_used=battery_used,
            grid_import=grid_import,
            wasted_solar=wasted_solar,
            grid_price=price,
        )

        self.hour += 1
        self.steps_taken += 1

        done = self.steps_taken >= self.episode_length

        # Sample the next hour's conditions.
        self.demand = self.sample_demand(self.hour)
        self.solar = self.sample_solar(self.hour)
        self.grid_price = self.get_grid_price(self.hour)

        next_state = self.get_state()

        info = {
            "hour": self.hour,
            "action": action,
            "action_name": self.actions[action],
            "demand": demand,
            "solar": solar,
            "grid_price": price,
            "battery_energy": self.battery_energy,
            "battery_soc": self.battery_energy / self.battery_capacity,
            "battery_used": battery_used,
            "grid_import": grid_import,
            "wasted_solar": wasted_solar,
        }

        info.update(reward_parts)

        return next_state, reward, done, info

    def get_state(self):
        """
        State vector for DQN.

        Values are normalised to help the neural network train.
        """
        battery_soc = self.battery_energy / self.battery_capacity

        demand_norm = self.demand / 4.0
        solar_norm = self.solar / 4.0
        price_norm = self.grid_price / 2.0

        # Time is cyclic, so hour 23 and hour 0 should be close.
        hour = self.hour % 24
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)

        return np.array(
            [
                battery_soc,
                demand_norm,
                solar_norm,
                price_norm,
                hour_sin,
                hour_cos,
            ],
            dtype=np.float32,
        )

    def sample_demand(self, hour):
        """
        Simple daily demand curve.

        Demand tends to rise in the morning and evening,
        with random noise added.
        """
        hour = hour % 24

        base = 1.2

        morning_peak = 0.8 * np.exp(-((hour - 8) ** 2) / 10)
        evening_peak = 1.2 * np.exp(-((hour - 19) ** 2) / 10)

        noise = self.rng.normal(0.0, self.demand_noise)

        demand = base + morning_peak + evening_peak + noise

        return float(np.clip(demand, 0.5, 3.5))

    def sample_solar(self, hour):
        """
        Simple daylight solar curve.

        Solar is zero at night, rises in the morning,
        peaks around midday, and falls in the evening.
        """
        hour = hour % 24

        if hour < 6 or hour > 18:
            return 0.0

        daylight_curve = np.sin((hour - 6) / 12 * np.pi)

        cloud_noise = self.rng.normal(1.0, self.solar_noise)
        cloud_noise = np.clip(cloud_noise, 0.4, 1.2)

        solar = 3.0 * daylight_curve * cloud_noise * self.solar_scale

        return float(np.clip(solar, 0.0, 3.5))

    def get_grid_price(self, hour):
        """
        Basic time-of-use price.

        Grid energy is more expensive during evening peak.
        """
        hour = hour % 24

        if 17 <= hour <= 21:
            return 1.5  # peak price
        elif 0 <= hour < 6:
            return 0.8  # off-peak price
        else:
            return 1.0  # normal price

    def get_grid_import(self, remaining_demand):
        """
        Grid import is limited.
        Anything beyond this becomes unmet demand.
        """
        return min(self.max_grid_import, max(0.0, remaining_demand))

    def calculate_reward(
        self,
        demand,
        solar,
        battery_used,
        grid_import,
        wasted_solar,
        grid_price,
    ):
        """
        Same basic idea as the discrete environment.

        We punish:
        - unmet demand heavily
        - grid import
        - wasted solar
        - battery use lightly

        In this version, grid import is also weighted by price.
        """
        unmet_demand = max(0.0, demand - solar - battery_used - grid_import)

        unmet_penalty = -10.0 * unmet_demand
        grid_penalty = -1.0 * grid_price * grid_import
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

    def state_size(self):
        return 6

    def num_actions(self):
        return len(self.actions)