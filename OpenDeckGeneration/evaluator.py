from typing import Optional
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error, r2_score


OUTPUT_FEATURES = ['deg_CmpBst_s_mapEff_in', 'deg_CmpBst_s_mapWc_in', 'deg_CmpFan_s_mapEff_in', 
                   'deg_CmpFan_s_mapWc_in', 'deg_CmpH_s_mapEff_in', 'deg_CmpH_s_mapWc_in', 
                   'deg_TrbH_s_mapEff_in', 'deg_TrbH_s_mapWc_in', 'deg_TrbL_s_mapEff_in', 
                   'deg_TrbL_s_mapWc_in']

def normalized_abs_error(y_true, y_pred, eps=1e-8):
    return np.abs(y_true - y_pred) / (np.abs(y_true) + eps)
    
def standardized_abs_error(y_true, y_pred, multioutput="average"):    
    std = np.std(y_true, axis=0)
    standard_error = np.abs(y_true - y_pred) / std
    metric = np.mean(standard_error, axis=0)
    if multioutput == "average":
        return metric
    else:
        return standard_error

def SMAPE(y_true, y_pred, multioutput="raw_outputs"):
    SAPE =  (
        2 * np.abs(y_pred - y_true) /
        (np.abs(y_true) + np.abs(y_pred))
    )
    if multioutput == "raw_outputs":
        return SAPE
    else:
        return np.mean(SAPE) * 100.
    
def NMSE(y_true, y_pred, multioutput="average"):
    nmse = ((y_true - y_pred)**2).mean(axis=0) / y_true.var(axis=0)
    overall_nmse = nmse.mean()
    if multioutput == "average":
        return overall_nmse
    else:
        return nmse
    
def NRMSE(y_true, y_pred, multioutput="average"):
    nrmse = np.sqrt(((y_true - y_pred)**2).mean(axis=0)) / y_true.std(axis=0)
    overall_nrmse = nrmse.mean()
    if multioutput == "average":
        return overall_nrmse
    else:
        return nrmse
    
def PEARSON_PER_VAR(y_true, y_pred):
    stat_per_var = []
    for i in range(y_true.shape[1]):
        pearson_stat = pearsonr(y_true[:,i], y_pred[:,i]).statistic
        stat_per_var.append(pearson_stat)
    return np.hstack(stat_per_var)


def evaluate_predictions(y_true, y_pred, metrics: dict = {}):
    if not(metrics):
        metrics["MAE"] = []
        metrics["mae_per_var"] = []
        metrics["r2"] = []
        metrics["r2_per_var"] = []
        metrics["STD_MAE"] = []
        metrics["STD_MAE_OBS"] = []
        metrics["SAPE"] = []
        metrics["SMAPE"] = []
        metrics["MAPE"] = []
        metrics["RMSE"] = []
        metrics["rmse_per_var"] = []
        metrics["NRMSE"] = []
        metrics["nrmse_per_var"] = []
        metrics["pearson_stat"] = []
        metrics["pearson_per_var"] = []
        metrics["pearson_pval"] = []
        metrics["spearman"] = []
        
    mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
    mae_per_var = mean_absolute_error(y_true=y_true, y_pred=y_pred, multioutput="raw_values")
    r2 = r2_score(y_true=y_true, y_pred=y_pred)
    r2_per_var = r2_score(y_true=y_true, y_pred=y_pred, multioutput="raw_values")
    mape = mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred)
    smape = SMAPE(y_true=y_true, y_pred=y_pred, multioutput="average")
    sape = SMAPE(y_true=y_true, y_pred=y_pred, multioutput="raw_outputs")
    rmse = root_mean_squared_error(y_true=y_true, y_pred=y_pred)
    rmse_per_var = root_mean_squared_error(y_true=y_true, y_pred=y_pred, multioutput="raw_values")
    nrmse = NRMSE(y_true=y_true, y_pred=y_pred)
    nrmse_per_var = NRMSE(y_true=y_true, y_pred=y_pred, multioutput="raw_values")
    pearson = pearsonr(y_true.flatten(), y_pred.flatten())
    pearson_per_var = pearsonr(y_true, y_pred, axis=0).statistic
    spearman = spearmanr(y_true.flatten(), y_pred.flatten()).statistic
    
    metrics["MAE"].append(mae)
    metrics["mae_per_var"].append(mae_per_var)
    metrics["r2"].append(r2)
    metrics["r2_per_var"].append(r2_per_var)
    metrics["MAPE"].append(mape)
    metrics["SMAPE"].append(smape)
    metrics["SAPE"].append(sape)
    metrics["RMSE"].append(rmse)
    metrics["rmse_per_var"].append(rmse_per_var)
    metrics["NRMSE"].append(nrmse)
    metrics["nrmse_per_var"].append(nrmse_per_var)
    metrics["pearson_stat"].append(pearson.statistic)
    metrics["pearson_pval"].append(pearson.pvalue)
    metrics["pearson_per_var"].append(pearson_per_var)
    metrics["spearman"].append(spearman)
    
    return metrics

def extract_metrics_for_table(metrics: dict, 
                              required_metrics: list=["SAPE", "rmse_per_var", "pearson_per_var"],
                              show_all_metrics: bool=False,
                              output_features: list=OUTPUT_FEATURES,
                              save_path: Optional[str]=None):
    metric_dict = dict()
    # print(metrics_per_category[cat].keys())
    if show_all_metrics:
        for metric, array in metrics.items():
            print(f"Selected metric: {metric}")
            print(f"Mean +/- Std : {np.array(array).mean()} +/- {np.array(array).std()}")        
    else:
        for metric in required_metrics:
            metric_dict[metric] = {}
            array = metrics[metric]
            array = np.vstack(array)
            array_mean = array.mean(axis=0)
            array_std = array.std(axis=0)
            for var, mean, std in zip(output_features, array_mean, array_std):
                if (var in ["empty_distance", "total_distance", "revenue"]) and (metric=="mae_per_var"):
                    metric_dict[metric][var] = f"{(mean/1e3):.4f} $\pm$ {(std/1e3):.4f}"
                else:
                    metric_dict[metric][var] = f"{mean:.4f} $\pm$ {std:.4f}"
                    
    if save_path is not None:
        pd.DataFrame.from_dict(metric_dict).to_csv(save_path)
    return metric_dict