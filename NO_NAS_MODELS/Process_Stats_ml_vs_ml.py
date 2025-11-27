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
                                df = pd.read_csv("Models/ml_vs_ml/gnn/Execution_" + str(exe) + "/Models/Graph_Neural_Training_metrics.csv")
                            elif ml_tech == "MLP":
                                df = pd.read_csv("Models/ml_vs_ml/MLP/Execution_" + str(exe) + "/Models/MLP_Training_metrics.csv")
                            elif ml_tech == "xgboost":
                                df = pd.read_csv("Models/ml_vs_ml/xgboost/Execution_" + str(exe) + "/Models/XGB_Training_metrics.csv")
                            for row in df.itertuples():
                                if row.node_size!='full':
                                    if (int(row.node_size)==node_size) & (row.model_type==feature_model):
                                        rmse_values.append(row.rmse)
                                        mape_values.append(row.mape)
                                else:
                                    if (row.node_size==node_size) & (row.model_type==feature_model):
                                        rmse_values.append(row.rmse)
                                        mape_values.append(row.mape)
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
            temp = list(map(lambda x: "All" if x == "full" else x, data_node_size))
            data_node_size = temp
            temp = list(map(lambda x: "QUBO" if x == "Q_values" else x, data_feature))
            temp = list(map(lambda x: "Ansatz" if x == "Circuit" else x, temp))
            temp = list(map(lambda x: "Padding" if x == "Q_values_full_model" else x, temp))
            data_feature = temp
            temp = list(map(lambda x: "GNN" if x == "gnn" else x, data_model))
            temp = list(map(lambda x: "XGBoost" if x == "xgboost" else x, temp))
            data_model = temp
            data={'Size': data_node_size,'Features':data_feature,'Model':data_model,'Best_RMSE':data_best_rmse, 'Worst_RMSE': data_worst_rmse,'Median_RMSE': data_rmse_median, 'IQR_RMSE': data_rmse_iqr,
                  'Best_MAPE':data_best_mape, 'Worst_MAPE': data_worst_mape,'Median_MAPE': data_mape_median, 'IQR_MAPE': data_mape_iqr}
    return data


if __name__ == "__main__":
  results = process_stats()
  results_formatted = pd.DataFrame(results)
  results_formatted.to_csv('Models/ml_vs_ml/statistics_ml_vs_ml.csv')
