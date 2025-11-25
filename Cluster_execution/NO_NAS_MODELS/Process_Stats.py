import numpy as np
import pandas as pd
# Import statistics Library
import statistics
import numpy as np
from scipy import stats

def process_stats():
    data_feature = []
    data_node_size = []
    data_model = []
    data_rmse_median = []
    data_rmse_iqr = []
    data_mape_median = []
    data_mape_iqr = []
    data_best_mape = []
    data_worst_mape = []
    data_best_rmse = []
    data_worst_rmse = []
#    for node_size in [10]:
#       for feature_model in ["Q_values"]:
#           for ml_tech in ["gnn"]:
    for node_size in [10, 12, 15, 20, 25, "full"]:
        for feature_model in ["Q_values", "Circuit", "Q_values_full_model"]:
            for ml_tech in ["gnn", "MLP", "xgboost"]:
                rmse_values = []
                mape_values = []
                if not((ml_tech == "gnn") & ((feature_model=="Circuit") or (feature_model=="Q_values_full_model"))):
                    if not (((ml_tech == "MLP") or (ml_tech == "xgboost")) & (feature_model == "Q_values_full_model") & (node_size == "full")):
                        for exe in range(1, 31):
                            if ml_tech == "gnn":
                                df = pd.read_csv("Models/gnn/Execution_" + str(exe) + "/Models/Graph_Neural_Training_metrics.csv")
                            elif ml_tech == "MLP":
                                df = pd.read_csv("Models/MLP/Execution_" + str(exe) + "/Models/MLP_Training_metrics.csv")
                            elif ml_tech == "xgboost":
                                df = pd.read_csv("Models/xgboost/Execution_" + str(exe) + "/Models/XGB_Training_metrics.csv")
                            for row in df.itertuples():
                                if row.node_size!='full':
                                    if (int(row.node_size)==node_size) & (row.model_type==feature_model):
                                        rmse_values.append(row.rmse)
                                        mape_values.append(row.mape)
                                elif (row.node_size==node_size) & (row.model_type==feature_model):
                                        rmse_values.append(row.rmse)
                                        mape_values.append(row.mape)
                        #print(rmse_values)
                        #print(mape_values)
                        # Print results
                        data_node_size.append(node_size)
                        data_feature.append(feature_model)
                        data_model.append(ml_tech)
                        data_rmse_median.append(np.median(rmse_values))
                        data_rmse_iqr.append(stats.iqr(rmse_values))
                        data_mape_median.append(np.median(mape_values))
                        data_mape_iqr.append(stats.iqr(mape_values))
                        data_best_rmse.append(min(rmse_values))
                        data_worst_rmse.append(max(rmse_values))
                        data_best_mape.append(min(mape_values))
                        data_worst_mape.append(max(mape_values))
            data={'Size': data_node_size,'Features':data_feature,'Model':data_model,'Best RMSE':data_best_rmse, 'Worst RMSE': data_worst_rmse,'Median RMSE': data_rmse_median, 'IQR RMS': data_rmse_iqr,
                  'Best MAPE':data_best_mape, 'Worst MAPE': data_worst_mape,'Median MAPE': data_mape_median, 'IQR MAPE': data_mape_iqr}
    return data


if __name__ == "__main__":
  results = process_stats()
  results_formatted = pd.DataFrame(results)
  print(results_formatted)
  results_formatted.to_csv('Models/statistics.csv')
