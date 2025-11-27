import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
plt.rcParams["font.family"] = "serif"




models_list=["MLP","XGBoost","GNN"]
df = pd.read_csv("Models/ml_vs_ml/statistics_ml_vs_ml.csv")

for feature_model in ["QUBO", "Ansatz", "Padding"]:
    if feature_model == "Padding":
        graph_size = [10, 12, 15, 20, 25]
    else:
        graph_size = [10, 12, 15, 20, 25,"All"]
    for metric in ["RMSE", "MAPE"]:
        barWidth = 0.2
        fig = plt.figure(figsize=(15, 5))
        fig.tight_layout()
        ax = plt.subplot(111)
        plt.rcParams['font.size'] = 9
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        num_bars = 0
        aux_num_bars = 0
        step = 0
        for ml_tech in models_list:
            data_iqr = []
            data_median = []
            if not ((ml_tech == "GNN") & ((feature_model == "Ansatz") or (feature_model == "Padding"))):
                for node_size in graph_size:
                    if not (((ml_tech == "MLP") or (ml_tech == "XGBoost")) & (feature_model == "Padding") & (node_size == "All")):
                        num_bars += 1
                        for row in df.itertuples():
                            if row.Size != 'All':
                                if (int(row.Size)== node_size) & (row.Features == feature_model) & (row.Model == ml_tech):
                                    if metric=="RMSE":
                                        data_median.append(row.Median_RMSE)
                                        data_iqr.append(row.IQR_RMSE)
                                    elif metric=="MAPE":
                                        data_median.append(row.Median_MAPE)
                                        data_iqr.append(row.IQR_MAPE)
                            else:
                                if (row.Size == node_size) & (row.Features == feature_model) & (row.Model == ml_tech):
                                    if metric == "RMSE":
                                        data_median.append(row.Median_RMSE)
                                        data_iqr.append(row.IQR_RMSE)
                                    elif metric == "MAPE":
                                        data_median.append(row.Median_MAPE)
                                        data_iqr.append(row.IQR_MAPE)

                        if ml_tech=="MLP":
                            aux_num_bars = num_bars
                            step = 0
                        else:
                            step = num_bars/aux_num_bars - 1

                graph_size_ticks = np.arange(len(graph_size))
                if ml_tech == "GNN":
                    color='orange'
                elif ml_tech == "MLP":
                    color='blue'
                elif ml_tech == "XGBoost":
                    color='green'
            if not ((ml_tech == "GNN") & ((feature_model == "Ansatz") or (feature_model == "Padding"))):
                plt.bar(graph_size_ticks+(step*barWidth), data_median,color =color, width = barWidth,label=f'{ml_tech.upper()}')
                plt.errorbar(graph_size_ticks+(step*barWidth), data_median,yerr=data_iqr,fmt="o",color ="r",label=f'{ml_tech.upper()} IQR')

        if (feature_model=="QUBO") & (metric=="MAPE"):
            ax.set_yscale('log')
            plt.xlabel("Graph Size",weight='bold',fontsize="20")
            plt.ylabel(f'Median {metric} (log scale)',weight='bold',fontsize="20")
        else:
            plt.xlabel("Graph Size",weight='bold',fontsize="20")
            plt.ylabel(f'Median {metric}',weight='bold',fontsize="20")
        ax.set_xticks(graph_size_ticks + 0.2)
        ax.set_xticklabels(graph_size)
        plt.legend(frameon=False)
        ax.spines[['right', 'top']].set_visible(False)
        ax.legend(fontsize="15", frameon=False,loc="upper left")

        """
        for i in range(len(values_1)):
            plt.text(i-0.2,values_1[i]+2,values_1[i])
        for i in range(len(values_2)):
            plt.text(i+2+0.3,values_2[i]+2,values_2[i])
        """

        plt.savefig(f'Figures/ml_vs_ml/{feature_model}_{metric}.pdf', format="pdf", bbox_inches="tight")
        plt.show()