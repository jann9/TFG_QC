import numpy as np
import pandas as pd
# Import statistics Library
import statistics
import numpy as np
from scipy import stats

def process_stats():
    data_node_size = []
    data_model = []

    data_time_median = []
    data_time_best = []
    data_time_worst = []
    data_time_iqr = []

    data_speedup_median = []
    data_speedup_best = []
    data_speedup_worst = []
    data_speedup_iqr = []

    data_Cut_median = []
    data_Cut_best = []
    data_Cut_worst = []
    data_Cut_iqr = []

    data_Optimal_Cut_median = []
    data_Optimal_Cut_best = []
    data_Optimal_Cut_worst = []
    data_Optimal_Cut_iqr = []


    data_cut_ratio_median = []
    data_cut_ratio_best = []
    data_cut_ratio_worst = []
    data_cut_ratio_iqr = []

    data_tradeoff_median = []
    data_tradeoff_best = []
    data_tradeoff_worst = []
    data_tradeoff_iqr = []

    for node_size in [10, 12, 15, 20, 25]:
        for ml_tech in ["Vanilla","xgboost", "MLP"]:
            data_node_size.append(node_size)
            data_model.append(ml_tech)
            data_time = []
            data_Speedup = []
            data_Cut = []
            data_Optimal_Cut = []
            data_cut_ratio = []
            data_tradeoff = []
            for exe in range(1, 31):
                df = pd.read_csv("Models/qaoa_vs_ml/execution_" + str(exe) + "/vanila_vs_ml.csv")
                for row in df.itertuples():
                    if (row.Approach==ml_tech) & (int(row.Size)==node_size):
                        data_time.append(row.Time)
                        data_Speedup.append(row.Time_Speed_Up)
                        data_Cut.append(row.Cut)
                        data_Optimal_Cut.append(row.Optimal_Cut)
                        data_cut_ratio.append(row.Cut_Ratio)
                        data_tradeoff.append(row.Tradeoff)

            data_time_median.append(np.median(data_time))
            data_time_best.append(min(data_time))
            data_time_worst.append(max(data_time))
            data_time_iqr.append(stats.iqr(data_time))

            data_speedup_median.append(np.median(data_Speedup))
            data_speedup_best.append(max(data_Speedup))
            data_speedup_worst.append(min(data_Speedup))
            data_speedup_iqr.append(stats.iqr(data_Speedup))

            data_Cut_median.append(np.median(data_Cut))
            data_Cut_best.append(max(data_Cut))
            data_Cut_worst.append(min(data_Cut))
            data_Cut_iqr.append(stats.iqr(data_Cut))

            data_Optimal_Cut_median.append(np.median(data_Optimal_Cut))
            data_Optimal_Cut_best.append(max(data_Optimal_Cut))
            data_Optimal_Cut_worst.append(min(data_Optimal_Cut))
            data_Optimal_Cut_iqr.append(stats.iqr(data_Optimal_Cut))

            data_cut_ratio_median.append(np.median(data_cut_ratio))
            data_cut_ratio_best.append(max(data_cut_ratio))
            data_cut_ratio_worst.append(min(data_cut_ratio))
            data_cut_ratio_iqr.append(stats.iqr(data_cut_ratio))

            data_tradeoff_median.append(np.median(data_tradeoff))
            data_tradeoff_best.append(max(data_tradeoff))
            data_tradeoff_worst.append(min(data_tradeoff))
            data_tradeoff_iqr.append(stats.iqr(data_tradeoff))

    data={'Size': data_node_size, 'Model': data_model, 'Best_Time':data_time_best, 'Worst_Time':data_time_worst, 'Median_Time':data_time_median,'IQR_Time':data_time_iqr,
    'Best_Time_Speedup':data_speedup_best, 'Worst_Time_Speedup':data_speedup_worst, 'Median_Time_Speedup':data_speedup_median,'IQR_Time_Speedup':data_speedup_iqr,
    'Best_Cut':data_Cut_best, 'Worst_Cut':data_Cut_worst, 'Median_Cut':data_Cut_median,'IQR_Cut':data_Cut_iqr,
    'Best_Cut_Ratio':data_cut_ratio_best, 'Worst_Cut_Ratio':data_cut_ratio_worst, 'Median_Cut_Ratio':data_cut_ratio_median,'IQR_Cut_Ratio':data_cut_ratio_iqr,
    'Best_Tradeoff':data_tradeoff_best, 'Worst_Tradeoff':data_tradeoff_worst, 'Median_Tradeoff':data_tradeoff_median, 'IQR_Tradeoff':data_tradeoff_iqr, 'Optimal_Cut':data_Optimal_Cut_median}
    return data


if __name__ == "__main__":
  results = process_stats()
  results_formatted = pd.DataFrame(results)
  results_formatted.to_csv('Models/qaoa_vs_ml/statistics_vanilla_vs_ml.csv')
