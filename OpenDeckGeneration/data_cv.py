import os
import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import  ShuffleSplit

def get_package_root():
    """
    Returns the absolute path to the root of the package.
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def create_kfold_splits(
        n_sequences: int,
        k_folds: int=5,
        train_ratio: float=0.7,
        val_ratio: float=0.15,
        test_ratio: float=0.15,
        random_state: int=42):
    """
    Generator yielding (train_set, val_set, test_set) for each fold.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    indices = np.arange(n_sequences)

    ss = ShuffleSplit(n_splits=k_folds, 
                      test_size=1-train_ratio, 
                      random_state=random_state)

    for train_idx, valtest_idx in ss.split(indices):

        # Split val/test inside the fold
        total_vt = len(valtest_idx)
        val_size = int(total_vt * val_ratio / (val_ratio + test_ratio))

        val_idx = valtest_idx[:val_size]
        test_idx = valtest_idx[val_size:]
        
        yield train_idx, val_idx, test_idx



def create_flatten_datasets(sequences: list,
                            train_idx: NDArray,
                            val_idx: NDArray,
                            test_idx: NDArray,
                            scale_inputs: bool=False,
                            scale_outputs: bool=False,
                            shuffle_train: bool=True):
    scaler_input = None
    scaler_output = None
    # inputs = np.array([sequence[:,1:8] for sequence in sequences], dtype="O")
    # outputs = np.array([sequence[:,-13:-3] for sequence in sequences], dtype="O")
    input_indices = [*list(range(14,21)), *list(range(27,34)), *list(range(40,47)), *list(range(53,60))]
    inputs = np.array([sequence[:,input_indices] for sequence in sequences], dtype="O")
    outputs = np.array([sequence[:,:10] for sequence in sequences], dtype="O")
    
    train_sensors = np.asarray(np.vstack(inputs[train_idx]), dtype=float)
    train_indicators = np.asarray(np.vstack(outputs[train_idx]), dtype=float)
    val_sensors = np.asarray(np.vstack(inputs[val_idx]), dtype=float)
    val_indicators = np.asarray(np.vstack(outputs[val_idx]), dtype=float)
    test_sensors = np.asarray(np.vstack(inputs[test_idx]), dtype=float)
    test_indicators = np.asarray(np.vstack(outputs[test_idx]), dtype=float)
    
    if shuffle_train:
        shuffle_indices = np.arange(len(train_sensors))
        np.random.shuffle(shuffle_indices)
        train_sensors = train_sensors[shuffle_indices]
        train_indicators = train_indicators[shuffle_indices]
    
    if scale_inputs:
        scaler_input = MinMaxScaler()
        train_sensors = scaler_input.fit_transform(train_sensors)
        val_sensors = scaler_input.transform(val_sensors)
        test_sensors = scaler_input.transform(test_sensors)
        
    if scale_outputs:
        scaler_output = MinMaxScaler()
        train_indicators = scaler_output.fit_transform(train_indicators)
        val_indicators = scaler_output.transform(val_indicators)
        test_indicators = scaler_output.transform(test_indicators)
        
    train_dataset = (train_sensors, train_indicators) 
    val_dataset = (val_sensors, val_indicators)
    test_dataset = (test_sensors, test_indicators) 
    
    return train_dataset, val_dataset, test_dataset, scaler_output

if __name__ == "__main__":
    import os
    import pandas as pd
    
    file_name = "sequences_with_info.csv"
    data_path = os.path.join(get_package_root(), "data", file_name)
    data = pd.read_csv(data_path)
    input_features = list(data.columns[2:9])
    output_features = list(data.columns[-13:-3])
        
    sequences = (
        data.groupby("sequence_id", sort=False)[data.columns[1:]]
            .apply(np.asarray)   # each group (DataFrame) -> ndarray
            .tolist()
    )


    for train_idx, val_idx, test_idx in create_kfold_splits(
            n_sequences=len(sequences),
            k_folds=5,
            train_ratio=0.7,
            val_ratio=0.1,
            test_ratio=0.2):
        
        train_dataset, val_dataset, test_dataset, scaler = create_flatten_datasets(sequences, 
                                                                                   train_idx, 
                                                                                   val_idx, 
                                                                                   test_idx,
                                                                                   scale_inputs=True,
                                                                                   shuffle_train=True)
        
        train_sensors, train_indicators = train_dataset
        val_sensors, val_indicators = val_dataset
        test_sensors, test_indicators = test_dataset
        
        print(train_sensors.shape)
        print(val_sensors.shape)
        print(test_sensors.shape)
        break