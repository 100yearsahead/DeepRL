# INM707 Deep RL Project Notes

## Project idea

The project models a simplified solar microgrid with battery storage. The agent controls the battery by choosing whether to charge, idle, or discharge at each timestep.

The goal is to meet demand while reducing grid import, avoiding wasted solar generation, and limiting unnecessary battery cycling.

## Basic environment

State:
- battery_level in {0, ..., 10}
- demand_regime in {low, medium, high}
- solar_regime in {low, medium, high}
- time_bin in {night, morning, midday, evening}

Actions:
- charge
- idle
- discharge

Why this is suitable:
- finite state space
- discrete action space
- interpretable
- suitable for tabular Q-learning

## Markov assumption

The next state distribution depends on the current battery level, demand regime, solar regime, time-of-day, and action. This is an approximation of a real microgrid, but it is sufficient for a simplified MDP because the relevant variables for the next control decision are included in the state.

# INM707 Deep Reinforcement Learning Project Notes

## Project idea

The project models a simplified solar microgrid with battery storage.

At each step, the agent sees:

- battery level
- demand level
- solar generation level
- time of day

The agent can choose one of three actions:

- charge
- idle
- discharge

The goal is to learn a sensible battery policy that meets demand while reducing grid import, avoiding wasted solar, and limiting unnecessary battery cycling.

---

## Basic environment

The state is written as:

```python
(battery_level, demand_regime, solar_regime, time_bin)
```

Where:

- `battery_level` is from 0 to 10
- `demand_regime`: 0 = low, 1 = medium, 2 = high
- `solar_regime`: 0 = low, 1 = medium, 2 = high
- `time_bin`: 0 = night, 1 = morning, 2 = midday, 3 = evening

The actions are:

- `0 = charge`
- `1 = idle`
- `2 = discharge`

This setup is suitable for tabular Q-learning because the state space and action space are both finite.

---

## Environment test 1: random action sanity check

Before adding Q-learning, I wanted to check that the environment itself actually behaves in a sensible way. For this first test I just used random actions, so the goal was not performance yet. I only wanted to see whether the battery, demand, solar, grid import, and reward values changed in a way that made sense.

For example:

```python
(5, 1, 0, 0)
```

means the battery is at level 5, demand is medium, solar is low, and the time is night.

### Output from the test

```text
Initial state: (5, 1, 0, 0)
Readable: {'battery_level': 5, 'demand': 'medium', 'solar': 'low', 'time': 'night'}

Step: 1
Action: charge
Next state: (5, 2, 0, 0)
Readable: {'battery_level': 5, 'demand': 'high', 'solar': 'low', 'time': 'night'}
Reward: -2.0
Info: {'hour': 1, 'action': np.int64(0), 'action_name': 'charge', 'demand': 2.0, 'solar': 0.0, 'battery_level': 5, 'battery_used': 0.0, 'grid_import': 2.0, 'wasted_solar': 0.0, 'unmet_demand': 0.0, 'unmet_penalty': -0.0, 'grid_penalty': -2.0, 'solar_waste_penalty': -0.0, 'battery_penalty': -0.0}

Step: 2
Action: charge
Next state: (5, 1, 0, 0)
Readable: {'battery_level': 5, 'demand': 'medium', 'solar': 'low', 'time': 'night'}
Reward: -3.0
Info: {'hour': 2, 'action': np.int64(0), 'action_name': 'charge', 'demand': 3.0, 'solar': 0.0, 'battery_level': 5, 'battery_used': 0.0, 'grid_import': 3.0, 'wasted_solar': 0.0, 'unmet_demand': 0.0, 'unmet_penalty': -0.0, 'grid_penalty': -3.0, 'solar_waste_penalty': -0.0, 'battery_penalty': -0.0}

Step: 3
Action: discharge
Next state: (4, 0, 0, 0)
Readable: {'battery_level': 4, 'demand': 'low', 'solar': 'low', 'time': 'night'}
Reward: -1.1
Info: {'hour': 3, 'action': np.int64(2), 'action_name': 'discharge', 'demand': 2.0, 'solar': 0.0, 'battery_level': 4, 'battery_used': 1.0, 'grid_import': 1.0, 'wasted_solar': 0.0, 'unmet_demand': 0.0, 'unmet_penalty': -0.0, 'grid_penalty': -1.0, 'solar_waste_penalty': -0.0, 'battery_penalty': -0.1}

Step: 4
Action: idle
Next state: (4, 1, 0, 0)
Readable: {'battery_level': 4, 'demand': 'medium', 'solar': 'low', 'time': 'night'}
Reward: -1.0
Info: {'hour': 4, 'action': np.int64(1), 'action_name': 'idle', 'demand': 1.0, 'solar': 0.0, 'battery_level': 4, 'battery_used': 0.0, 'grid_import': 1.0, 'wasted_solar': 0.0, 'unmet_demand': 0.0, 'unmet_penalty': -0.0, 'grid_penalty': -1.0, 'solar_waste_penalty': -0.0, 'battery_penalty': -0.0}

Step: 5
Action: idle
Next state: (4, 0, 0, 0)
Readable: {'battery_level': 4, 'demand': 'low', 'solar': 'low', 'time': 'night'}
Reward: -2.0
Info: {'hour': 5, 'action': np.int64(1), 'action_name': 'idle', 'demand': 2.0, 'solar': 0.0, 'battery_level': 4, 'battery_used': 0.0, 'grid_import': 2.0, 'wasted_solar': 0.0, 'unmet_demand': 0.0, 'unmet_penalty': -0.0, 'grid_penalty': -2.0, 'solar_waste_penalty': -0.0, 'battery_penalty': -0.0}
```

### What I noticed

The environment seems to be working as expected.

The starting state was:

```python
(5, 1, 0, 0)
```

This makes sense because the battery starts half full, demand is medium, solar is low, and the episode begins at night.

In the first two steps, the random action was `charge`. Since it was night and solar generation was zero, there was no surplus solar to charge the battery with. So the battery stayed at level 5. The grid had to cover the demand, which is why the rewards were `-2.0` and `-3.0`.

In step 3, the action was `discharge`. This reduced the battery from 5 to 4. The battery supplied 1 unit of energy, so grid import dropped to 1.0. The reward was `-1.1`, which comes from the grid penalty plus a small battery-use penalty. This is good because it shows that discharging can help reduce grid dependence, but it is not completely free.

In steps 4 and 5, the agent idled. Since solar was still zero, the grid covered the demand. The reward was just the grid import penalty.

One thing I noticed is that `unmet_demand` stays at zero in this version. That is because the grid is currently allowed to cover all demand not met by solar or the battery. This is fine for the first version, but later it may be useful to add a maximum grid import limit so unmet demand can actually happen. That would make the load-balancing problem more interesting.

### Small clean-up

The action values were printing as `np.int64(0)` instead of just `0`. This is because the random actions were sampled using NumPy. It does not affect the logic, but it makes the logs look messy.

I fixed this by adding:

```python
action = int(action)
```

at the start of the `step()` method.

### Conclusion

This first test confirms that the basic environment is behaving sensibly. The battery changes when it should, grid import reacts to demand and solar, and the reward values reflect the cost of using the grid, wasting solar, and using the battery.

## Random policy baseline

After the initial step-by-step sanity check, I ran a full 24-step episode using a random policy. The agent selected between `charge`, `idle`, and `discharge` uniformly at random.

The purpose of this test was to create a weak baseline before implementing Q-learning. I want the learned agent to achieve a better total reward and, more importantly, reduce grid import by using the battery more sensibly.

### Episode summary

```text
Random policy episode summary
-----------------------------
Total reward: -32.6
Total grid import: 31.0
Total wasted solar: 2.0
Total battery used: 6.0
Total unmet demand: 0.0
```

### Interpretation

The random policy performs poorly, which is expected. It does not understand when the battery should be saved or used.

One important pattern is that the random agent discharges the battery several times during the morning and midday, even when solar generation is already enough to cover demand. For example, around steps 11 and 12, the agent discharges even though grid import is already zero. This gives a small battery penalty without improving the energy balance.

By step 16, the battery has dropped to zero. This creates a problem later in the episode because the evening period has higher demand and lower solar generation. From steps 19 to 24, the battery remains empty, so the grid has to cover most of the demand. This leads to large grid import values and more negative rewards.

This baseline is useful because it shows what the learned policy should improve. A better policy should:
- avoid wasting battery when solar already covers demand;
- charge during solar surplus periods;
- preserve battery energy for evening/high-demand periods;
- reduce total grid import;
- achieve a less negative total reward.

At this stage, unmet demand is still zero because the grid can cover all remaining demand. Later, if I add a grid import limit, unmet demand will become a more meaningful reliability metric.


## Random policy baseline after adding grid limit

After adding a maximum grid import limit, I reran the random policy baseline for one 24-step episode.

This version is more interesting than the first test because the grid can no longer cover unlimited demand. The grid can only import up to 2 units per step. Therefore, if demand is high, solar is low, and the battery cannot help, unmet demand becomes possible.

### Episode summary

```text
Random policy episode summary
-----------------------------
Total reward: -77.6
Total grid import: 26.0
Total wasted solar: 2.0
Total battery used: 6.0
Total unmet demand: 5.0
```

### What changed

In the earlier version, unmet demand was always zero because the grid could cover all remaining demand. After limiting grid import, unmet demand appears in difficult states.

For example, in step 2:

```text
demand = 3.0
solar = 0.0
battery_used = 0.0
grid_import = 2.0
unmet_demand = 1.0
reward = -12.0
```

This makes sense. Demand was high, there was no solar, and the action did not use the battery. Since the grid could only supply 2 units, 1 unit of demand was unmet. Because unmet demand has a large penalty, the reward was much worse.

The same issue appears later in the evening when the battery is empty. In steps 21 to 24, demand is high, solar is zero, and the battery cannot supply energy. The grid limit means the system cannot fully meet demand, so unmet demand becomes positive and the rewards are strongly negative.

### Interpretation

This is a better load-balancing environment because the agent now has to manage a limited resource. Random behaviour wastes the battery earlier in the day and then has no stored energy left during the evening, when solar is low and demand is often high.

A learned policy should improve on this by:

- avoiding unnecessary discharge when solar already covers demand;
- charging when solar surplus is available;
- saving battery energy for evening or high-demand periods;
- reducing unmet demand;
- reducing grid import where possible;
- achieving a less negative total reward than the random policy.

This gives a clear baseline for Q-learning to beat.