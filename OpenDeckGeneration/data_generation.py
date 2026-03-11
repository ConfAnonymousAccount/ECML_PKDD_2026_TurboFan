import os
import argparse
import yaml
import uuid
from multiprocessing import Pool
from tqdm import tqdm
from typing import Optional
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from odsmr.sensors import HPC_Tout, HP_Nmech, HPC_Tin, LPT_Tin, Fuel_flow, HPC_Pout_st, LP_Nmech
from odsmr.predefined_flight_conditions import Cruise_DeckSMR, Takeoff_DeckSMR, Climb1_DeckSMR, Climb2_DeckSMR
from odsmr.generation_functions import decksmr_1forall
from odsmr.constants import ROOT_OPENDECK

ALL_CONTEXTS = [Cruise_DeckSMR, Takeoff_DeckSMR, Climb1_DeckSMR, Climb2_DeckSMR]
SENSOR_LABELS = [HPC_Tout.__name__, 
                 HP_Nmech.__name__, 
                 HPC_Tin.__name__, 
                 LPT_Tin.__name__, 
                 Fuel_flow.__name__, 
                 HPC_Pout_st.__name__, 
                 LP_Nmech.__name__]

def add_bounded_noise(x, gamma=0.02):
    delta = max(x) - min(x)
    noise = np.random.uniform(-gamma*delta, gamma*delta, size=x.shape)
    return x + noise

def add_scaled_noise(x, alpha=0.05):
    x = np.array(x, dtype=float)
    norm = (x - min(x)) / (max(x) - min(x))
    sigma = alpha * norm
    noise = np.random.normal(0, sigma, size=x.shape)
    return x + noise

def generate_data(trajectory_sequences: NDArray,
                  maintenance_occurances: NDArray,
                  speed_change_occurances: NDArray,
                  speed_strategy_selected: NDArray,
                  list_sensors: list, 
                  context: str="from_list", 
                  context_type: Optional[list]=ALL_CONTEXTS,
                  noise_type: Optional[str]=None,
                  noise_params: Optional[dict]=None,
                  save_path: Optional[str]=None,
                  file_name: str="sequences") -> list:
    """function to generate measurements based on config and degradation states sequences
    
    TODO: Add some verification on dimensionality of trajecotry sequences


    Parameters
    ----------
    trajectory_sequences : NDArray
        the trajectories generated using the functionalities in trajectoryGeneration.py script
    maintenance_occurrences : NDArray
        the maintenance occurrences corresponding to the generated sequences
    speed_change_occurances : NDArray
        the degradation speed change occurrences corresponding to the generated sequences
    speed_strategy_selected : NDArray
        speed strategies corresponding to the generated sequences. (slow, normal, fast)
    list_sensors : list
        the list of sensors (measures) that should be computed by the OpenDeckSMR solver.
    context : str, optional
        whether to generate randomly the context per sequence or select from the context type, by default "constant"
        - `from_list`: use the context type variable as a constant value for context
        - `random`: to randomize the context selection among Cruise_DeckSMR, Takeoff_DeckSMR, Climb1_DeckSMR, Climb2_DeckSMR
    context_type : list, optional
        the list from which the context should be selected, by default [Cruise_DeckSMR]
        It is used if the `context` parameter is set to `from_list`
    noise_type: str, optional
        the type of noise that should be added to the sensor values, by default None
        the None value means no noise to be added
        the current available options are : "Bounded" and "Scaled"
    noise_params: dict, optional
        the parameters related to the selected noise, by default None
        this is mandatory if the noise type is provided
    save_path : Optional[str], optional
        if indicated, the final array is saved
    file_name : str, optional
        The file name use to save the a csv file, by default "sequences"
    
    Returns
    -------
    list
        all the generated sequences

    Raises
    ------
    Exception
        _description_
    NotImplementedError
        _description_
    Exception
        _description_
    NotImplementedError
        _description_
    """    
    # TODO: add tqdm
    all_sequences = []
    for sequence, maintenance, speed_change, speed_strategy in tqdm(zip(trajectory_sequences, maintenance_occurances, speed_change_occurances, speed_strategy_selected)):
        if context == "random":
            context_type = [ALL_CONTEXTS[np.random.choice(4)]]
        elif (context == "from_list") and (context_type is None):
            raise Exception(f"context_type should not be None if context is from_list")
        # else:
        #     raise NotImplementedError(f"The indicated context {context} is not implemented. Try between 'random' or 'from_list'.")
        
        for context_ in context_type:
            info_df = decksmr_1forall(list_state_value=list(sequence),
                                      list_context=[context_],
                                      list_sensors=list_sensors,
                                      sim_root=ROOT_OPENDECK
                                      )
            info_df.insert(0, "timestep", np.arange(len(sequence)))
            info_df.insert(0, "sequence_id", uuid.uuid4())
            info_df["maintenance"] = maintenance
            info_df["speed_change"] = list(speed_change)
            info_df["speed_strategy"] = list(speed_strategy)
            
            if noise_type is not None:
                if noise_params is None:
                    raise Exception(f"the parameters for noise {noise_type} should be provided")
                if noise_type == "Bounded":
                    for label in SENSOR_LABELS:
                        info_df.loc[:, label] = add_bounded_noise(info_df.loc[:, label], **noise_params)
                elif noise_type == "Scaled":
                    for label in SENSOR_LABELS:
                        info_df.loc[:, label] = add_scaled_noise(info_df.loc[:, label], **noise_params)
                else:
                    raise NotImplementedError(f"the considered noise type {noise_type} is not implemented." \
                                            "Try one option among `Bounded` or `Scaled`")
                
            
            if save_path is not None:
                if not os.path.exists(save_path):
                    os.makedirs(save_path, exist_ok=True)
                file_path = os.path.join(save_path, f"{file_name}.csv")
                if not os.path.exists(file_path):

                    info_df.to_csv(file_path, index=False, header=True, mode="a")
                else:
                    info_df.to_csv(file_path, index=False, header=False, mode="a")
    
    return all_sequences


if __name__ == "__main__":
    from trajectoryGeneration import simulate_multiple_trajectories
    # python data_generation.py --read_from_config=True
    parser = argparse.ArgumentParser(prog="OpenDeckGeneration")
    parser.add_argument('--read_from_config', help="whether to read from config or use CLI", default="True", type=bool, required=True)
    parser.add_argument('--n_sequences', help="number of trajectories", default=1, type=int, required=False)
    parser.add_argument('--sequence_length', help="length of trajectories", default=1000, type=int, required=False)
    parser.add_argument('--init_val', help="Initialization strategy", default="zero", type=str, required=False)
    parser.add_argument('--speed_strategy', help="Degradation speed strategy (by default Random)", default="random", type=str, required=False)
    parser.add_argument('--change_speed_occurrence', help="Changing the speed frequency", default=100, type=int, required=False)
    parser.add_argument('--maintenance_interval', help="A random value is picked in this interval for each maintenance period", default="(200, 500)", type=str, required=False)
    parser.add_argument('--maintenance_coeff_range', help="A random value is picked in this interval for each sequence to retreive a percentage of previous health state", default="(0.6, 0.8)", type=str, required=False)
    parser.add_argument('--context', help="Context selection strategy", default="random", type=str, required=False)
    parser.add_argument('--context_type', help="A list of context types ", default="Cruise_DeckSMR", type=list, required=False)
    parser.add_argument('--noise_type', help="Noise type to be applied on measures", default="Bounded", type=str, required=False)
    parser.add_argument('--noise_params', help="Noise parameters to be used for the selected noise type", default="{'gamma': 0.03}", type=str, required=False)
    parser.add_argument('--save_path', help="The path where the generated data should be saved", default=".", type=str, required=False)
    parser.add_argument('--file_name', help="File name for saving", default="sequences", type=str, required=False)
    parser.add_argument('--n_jobs', help="Number of jobs for multi-processing", default=1, type=int, required=False)
    args = parser.parse_args()
    
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_file_dir, "config.yaml")
    
    if args.read_from_config:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        n_sequences = config.get("n_sequences", int(args.n_sequences))
        sequence_length = config.get("sequence_length", int(args.sequence_length))
        speed_strategy = config.get("speed_strategy", args.speed_strategy)
        init_value = config.get("init_value", args.init_val)
        change_speed_occurrence = config.get("change_speed_occurrence", int(args.change_speed_occurrence))
        maintenance_interval = eval(config.get("maintenance_interval", args.maintenance_interval))
        maintenance_coeff_range = eval(config.get("maintenance_coeff_range", args.maintenance_coeff_range))
        context = config.get("context", args.context)
        context_type = config.get("context_type", args.context_type)
        noise_type = config.get("noise_type", args.noise_type)
        noise_params = config.get("noise_params", eval(args.noise_params))
        save_path = config.get("save_path", args.save_path)
        file_name = config.get("file_name", args.file_name)
        n_jobs = config.get("n_jobs", int(args.n_jobs))
    else:
        n_sequences = int(args.n_sequences)
        sequence_length = int(args.sequence_length)
        speed_strategy = args.speed_strategy
        init_value = args.init_value
        change_speed_occurrence = int(args.change_speed_occurrence)
        maintenance_interval = eval(args.maintenance_interval)
        maintenance_coeff_range = eval(args.maintenance_coeff_range)
        context = args.context
        context_type = args.context_type
        noise_type = args.noise_type
        noise_params = eval(args.noise_params)
        save_path = args.save_path
        file_name = args.file_name
        n_jobs = args.n_jobs
    
    if context_type is not None:
        context_type = [eval(context_) for context_ in context_type]
        
    print("Parameters")
    print("----------")
    label_w = 28
    value_w = 20

    print(f"{'n_sequences:':<{label_w}}{str(n_sequences):<{value_w}}")
    print(f"{'sequence_length:':<{label_w}}{str(sequence_length):<{value_w}}")
    print(f"{'speed_strategy:':<{label_w}}{str(speed_strategy):<{value_w}}")
    print(f"{'init_value:':<{label_w}}{str(init_value):<{value_w}}")
    print(f"{'change_speed_occurrence:':<{label_w}}{str(change_speed_occurrence):<{value_w}}")
    print(f"{'maintenance_interval:':<{label_w}}{str(maintenance_interval):<{value_w}}")
    print(f"{'maintenance_coeff_range:':<{label_w}}{str(maintenance_coeff_range):<{value_w}}")
    print(f"{'context:':<{label_w}}{str(context):<{value_w}}")
    print(f"{'noise_type:':<{label_w}}{str(noise_type):<{value_w}}")
    print(f"{'noise_params:':<{label_w}}{str(noise_params):<{value_w}}")
    print(f"{'save_path:':<{label_w}}{str(save_path):<{value_w}}")
    print(f"{'file_name:':<{label_w}}{str(file_name):<{value_w}}")
    print(f"{'n_jobs:':<{label_w}}{str(n_jobs):<{value_w}}")
            
    list_sensors = [HPC_Tout(),
                    HP_Nmech(),
                    HPC_Tin(),
                    LPT_Tin(),
                    Fuel_flow(),
                    HPC_Pout_st(),
                    LP_Nmech()]
    
    results = simulate_multiple_trajectories(config_path=config_path,
                                             n_sequences=n_sequences, 
                                             sequence_length=sequence_length,
                                             speed_strategy=speed_strategy,
                                             init_value=init_value,
                                             change_speed_occurrence=change_speed_occurrence,
                                             maintenance_interval=maintenance_interval,
                                             maintenance_coeff_range=maintenance_coeff_range,
                                             seed=None)
    
    trajectories, maintenances, speed_occurrences, speed_strategies = results
    
    tasks = []
    n_tasks = int(n_sequences / n_jobs)
    for i in range(n_tasks):
        beg_idx = (i * n_tasks)
        end_idx = ((i+1) * n_tasks)
        # print(list(range(beg_idx, end_idx)))
        print(trajectories[beg_idx:end_idx].shape)
        tasks.append((trajectories[beg_idx:end_idx], maintenances[beg_idx:end_idx], speed_occurrences[beg_idx:end_idx], \
                      speed_strategies[beg_idx:end_idx], list_sensors, context, context_type, noise_type, noise_params, \
                      save_path, file_name))
    
    with Pool(n_jobs) as p:
        p.starmap(generate_data, tasks)
        
    
    # all_sequences = generate_data(trajectory_sequences=trajectories,
    #                               maintenance_occurances=maintenances,
    #                               speed_change_occurances=speed_occurrences,
    #                               speed_strategy_selected=speed_strategies,
    #                               list_sensors=list_sensors, 
    #                               context=context,
    #                               context_type=context_type,
    #                               noise_type=noise_type,
    #                               noise_params=noise_params,
    #                               save_path=save_path,
    #                               file_name=file_name
    #                               )
