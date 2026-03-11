import os
import yaml
import copy
from typing import Optional
import numpy as np
from numpy.typing import NDArray
import plotly.graph_objects as go
from odsmr.constants import STATE_BOUNDS

def simulate_degradation_trajectory(config: dict,
                                    sequence_length: int=1000, 
                                    speed_strategy: str="random", 
                                    init_value: str="zero",
                                    change_speed_occurrence: int=100,
                                    maintenance_interaval: tuple=(200,500),
                                    maintenance_coeff: float=0.8,
                                    seed=None) -> tuple:
    """Generate one degradation trajectory

    Parameters
    ----------
    config: dict
        a dictionary including all the configurations required for the trajectory generation
    sequence_length : int, optional
        the sequence length, by default 1000
    speed_strategy : str, optional
        the speed of degradation, by default "random"
        "random" means that the speed could change throughout the generation every `change_speed_occurrence`
        Other options are (slow, normal, fast). These could be changed via config file
    change_speed_occurrence: int, optional
        the parameters allows to change the frequence of sampling the speed strategy during the generation of trajectory
    init_value : str, optional
        the initial state value, by default "max"
        two options are available (`max`, `random`)
        - `max` means that we pick the higher bound of the interval for the initial state
        - `random` allows to picke a random value in the interval for each health state variable
    maintenance_interval: tuple, optional
        Various random values are picked in this interval, which designates various maintenance timings
        Each value designates the duration of the maintenance from the first time stamp or from a maintenance event
        --------------M---------------M--------------M--------------sequence_length
        0            230             640            910               1000
        init      0 + random    230 + random     640 + random        END  OF SEQ
    maintenance_coeff: float, optional
        the percentage of the previous health state to recover from. 
        Whether the initial point (zero) or the previous maintenance
          0  |*                         |         
             | *        * ------------->| recovering maintenance_coeff % of the previous health state
             |   *        *        *
             |    *         *         * 
       -0.05 |__________________________
    seed : int, optional
        the seed used for numpy.random, by default None
        
    Returns
    -------
    NDArray: a numpy array of a generated degradation sequence of dimension (`sequence_length`, `10`)
    """    
    
    division_factor = int(np.random.choice([sequence_length/factor for factor in config["speed_division"]["factors"]], 
                                           p=config["speed_division"]["distribution"]))
    
    speed_params = config["speed_params"]
    for speed_cat, speed_param in speed_params.items():
        for param, param_val in speed_param.items():
            param_val = float(param_val)
            param_val /= float(division_factor)
            speed_params[speed_cat][param] = param_val
            
    np.random.seed(seed)
    # set the maintenance times
    maintenance_times = np.random.choice(range(*(maintenance_interaval)), int(sequence_length/maintenance_interaval[0]))
    maintenance_times = list(np.cumsum(maintenance_times))
    tmp_maintenance_times = list([time for time in maintenance_times if time < sequence_length])
    tmp_maintenance_times = [el-1 for el in tmp_maintenance_times]
    maintenance_occurence = np.zeros(sequence_length, dtype=bool)
    maintenance_occurence[tmp_maintenance_times] = True
    speed_change_occurence = np.zeros((sequence_length, len(STATE_BOUNDS)), dtype=bool)
    speed_strategy_selected = np.zeros((sequence_length, len(STATE_BOUNDS)), dtype=str)
    
    trajectory = []
    for indicator_idx, (key, val) in enumerate(STATE_BOUNDS.items()):
        current_state = []
        min_bound, max_bound = val
        state_maintenance_times = copy.deepcopy(maintenance_times)
                
        if init_value == "max":
            # select the higher bound as the initial state value
            current_val = max_bound
        elif init_value == "random":
            # pick a random value for the initial state value
            current_val = np.random.uniform(min_bound, max_bound)
        elif init_value == "zero":
            current_val = 0.
        last_maintenance = 0
        for time_stamp in range(sequence_length):
            if ((time_stamp+1) % state_maintenance_times[0]) == 0:
                maintenance_time = state_maintenance_times.pop(0)
                maintenance_duration = maintenance_time - last_maintenance
                beg_ = (time_stamp+1) - maintenance_duration
                last_maintenance = maintenance_time -1
                current_val += maintenance_coeff * np.abs(current_state[-1] - current_state[beg_])
            # change the speed every change_speed_occurrence times
            if speed_strategy == "random":
                if not(time_stamp % change_speed_occurrence):
                    speed = np.random.choice(list(speed_params.keys()), 
                                            p=config["speed_probability_distribution"][key])
                    if time_stamp == 0:
                        previous_speed = speed
                    else:
                        if previous_speed != speed:
                            speed_change_occurence[time_stamp,indicator_idx] = True
                        previous_speed = speed
            else:
                speed = speed_strategy
            speed_strategy_selected[time_stamp,indicator_idx] = speed
            slope_mean = speed_params[speed]['mean_slope']
            slope_std = speed_params[speed]['std_slope']
            slope = np.random.normal(slope_mean, slope_std)
            # add a gaussian noise
            noise = np.random.normal(float(config["slope_noise"]["mean"]), 
                                     float(config["slope_noise"]["std"])/division_factor)
            current_val += slope + noise
            current_val = np.clip(current_val, min_bound, max_bound)
            current_state.append(current_val)
        trajectory.append(current_state)
    trajectory = np.array(trajectory).T
    filtered_trajectory = filter_trajectory(trajectory)
    traj_length = len(filtered_trajectory)
    return filtered_trajectory, maintenance_occurence[:traj_length], speed_change_occurence[:traj_length], speed_strategy_selected[:traj_length]

def simulate_multiple_trajectories(config_path: str,
                                   n_sequences: int=2,
                                   sequence_length: int=1000,
                                   speed_strategy: str="random",
                                   init_value="max",
                                   change_speed_occurrence: int=100,
                                   maintenance_interval: tuple=(200,500),
                                   maintenance_coeff_range: tuple=(0.6,0.8),
                                   seed: Optional[int]=None,
                                   save_path: Optional[str]=None) -> tuple:
    """simulates multiple trajectories
    
    Calls `n_sequences` times ``simulate_degradation_trajectory`` function. 

    Parameters
    ----------
    config: dict
        a dictionary including all the configurations required for the trajectory generation
    n_sequences : int, optional
        number of sequences to generate, by default 2
    sequence_length : int, optional
        the sequence length, by default 1000
    speed_strategy : str, optional
        the speed of degradation, by default "random"
        "random" means that the speed could change throughout the generation every `change_speed_occurrence`
        Other options are (slow, normal, fast). These could be changed via config file
    change_speed_occurrence: int, optional
        the parameters allows to change the frequence of sampling the speed strategy during the generation of trajectory
    init_value : str, optional
        the initial state value, by default "max"
        two options are available (`max`, `random`)
        - `max` means that we pick the higher bound of the interval for the initial state
        - `random` allows to picke a random value in the interval for each health state variable
    maintenance_interval: tuple, optional
        Various random values are picked in this interval, which designates various maintenance timings
        Each value designates the duration of the maintenance from the first time stamp or from a maintenance event
        --------------M---------------M--------------M--------------sequence_length
        0            230             640            910               1000
        init      0 + random    230 + random     640 + random        END  OF SEQ
    maintenance_coeff_range: tuple, optional
        the percentage of the previous health state to recover from.
        Whether the initial point (zero) or the previous maintenance.
        A random value is picked in this interval for each sequence. 
    seed : int, optional
        the seed used for numpy.random, by default None
    save_path: str, optional
        if indicated, the final sequances are saved using numpy saving capability (npz file)    
    
    Returns
    -------
    NDArray : an array of generated degradation sequences of dimension (`n_sequences`, `size`, `10`)
    """
    all_trajectories = []
    all_maintenance_occurances = []
    all_speed_change_occurances = []
    all_speed_strategy_selected = []
    for _ in range(n_sequences):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        maintenance_coefficient = np.random.uniform(*(maintenance_coeff_range))
        info = simulate_degradation_trajectory(config=config,
                                               sequence_length=sequence_length,
                                               speed_strategy=speed_strategy,
                                               init_value=init_value,
                                               change_speed_occurrence=change_speed_occurrence,
                                               maintenance_interaval=maintenance_interval,
                                               maintenance_coeff=maintenance_coefficient,
                                               seed=seed)
        trajectory, maintenance_occurances, speed_change_occurances, speed_strategy_selected = info
        all_trajectories.append(trajectory)
        all_maintenance_occurances.append(maintenance_occurances)
        all_speed_change_occurances.append(speed_change_occurances)
        all_speed_strategy_selected.append(speed_strategy_selected)
    
    all_trajectories = np.array(all_trajectories, dtype=object)
    all_maintenance_occurances = np.array(all_maintenance_occurances, dtype=object)
    all_speed_change_occurances = np.array(all_speed_change_occurances, dtype=object)
    all_speed_strategy_selected = np.array(all_speed_strategy_selected, dtype=object)
    
    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        file_path = os.path.join(save_path, "trajectories")
        np.savez(file_path, all_trajectories, allow_pickle=True)
            
    return all_trajectories, all_maintenance_occurances, all_speed_change_occurances, all_speed_strategy_selected

 
def filter_trajectory(trajectory: NDArray):
    """Filter the trajectories if they cross the bounds

    Parameters
    ----------
    trajectory : NDArray
        a trajectory including 10 indicators

    Returns
    -------
    NDArray
        Filtered trajectory
    """    
    index_list = []
    for idx, item in enumerate(STATE_BOUNDS):
        indices = np.where(trajectory[:, idx] <= STATE_BOUNDS[item][0])[0]
        if len(indices) > 0:
            index_list.append(indices[0])
            
    if len(index_list) > 0:
        index = min(index_list)
        return trajectory[:index, :]
    else:
        return trajectory


if __name__ == "__main__":
    
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_file_dir, "config.yaml") 

    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    # print(config)

    init_value = "zero"
    sequence_length=1000
    change_speed_occurrence = 100
    seed = np.random.randint(1000)
    
    results = simulate_degradation_trajectory(config=config,
                                              sequence_length=sequence_length,
                                              speed_strategy="random",
                                              init_value=init_value,
                                              change_speed_occurrence=change_speed_occurrence,
                                              maintenance_interaval=(200,500),
                                              maintenance_coeff=0.6,
                                              seed=seed)
    
    trajectory = results[0]
    maintenance_occurence = results[1] 
    speed_change_occurence = results[2]
    speed_strategy_selected = results[3]
    