import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
plt.rcParams["font.family"] = "serif"


version=2
df = pd.read_csv(f"Models/qaoa_vs_ml/V{version}/statistics_vanilla_vs_ml.csv")
graph_size = [10, 12, 15, 20, 25]
metrics=["Time","Time Speedup","Cut","Cut Ratio"]
for metric in metrics:
    if metric == "Cut":
        models_list = ["MLP", "xgboost", "Vanilla","Optimal"]
    else:
        models_list = ["MLP", "xgboost", "Vanilla"]
    step = 0
    num_bars = 0
    aux_num_bars = 0
    barWidth = 0.2
    fig = plt.figure(figsize=(15, 5))
    fig.tight_layout()
    ax = plt.subplot(111)
    plt.rcParams['font.size'] = 9
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    for ml_tech in models_list:
        if ml_tech == "Optimal":
           ml_tech_aux = "Vanilla"
        else:
           ml_tech_aux = ml_tech
        data_iqr = []
        data_median = []
        for node_size in graph_size:
            num_bars += 1
            for row in df.itertuples():
                if ((int(row.Size)== node_size) & (row.Model == ml_tech_aux)):
                    if metric == "Time":
                        data_median.append((float(row.Median_Time)))
                        data_iqr.append((float(row.IQR_Time)))
                    elif (metric == "Time Speedup") & (ml_tech != "Vanilla"):
                        data_median.append(float(row.Median_Time_Speedup))
                        data_iqr.append(float(row.IQR_Time_Speedup))
                    elif metric=="Cut":
                        if  ml_tech != "Optimal":
                            data_median.append(float(row.Median_Cut))
                            data_iqr.append(float(row.IQR_Cut))
                        else:
                            data_median.append(float(row.Optimal_Cut))
                    elif metric=="Cut Ratio":
                        data_median.append(float(row.Median_Cut_Ratio))
                        data_iqr.append(float(row.IQR_Cut_Ratio))

        if ml_tech_aux=="MLP":
            aux_num_bars = num_bars
            step = 0
        else:
            step = num_bars/aux_num_bars - 1

        graph_size_ticks = np.arange(len(graph_size))
        if (ml_tech_aux == "Vanilla") & (ml_tech == "Vanilla"):
            color='orange'
        elif ml_tech_aux == "MLP":
            color='blue'
        elif ml_tech_aux == "xgboost":
            color='green'
        elif ml_tech == "Optimal":
            color='k'
        if not((metric == "Time Speedup") & (ml_tech == "Vanilla")):
            plt.bar(graph_size_ticks+(step*barWidth), data_median,color =color, width = barWidth,label=f'{ml_tech.upper()}')
            if ml_tech != "Optimal":
                plt.errorbar(graph_size_ticks+(step*barWidth), data_median,yerr=data_iqr,fmt="o",color ="r",label=f'{ml_tech.upper()} IQR')


    if metric == "Time":
        ax.set_yscale('log')
        plt.xlabel("Graph Size", weight='bold', fontsize="20")
        plt.ylabel(f'Median {metric} (log scale)', weight='bold', fontsize="20")
    else:
        plt.xlabel("Graph Size", weight='bold', fontsize="20")
        plt.ylabel(f'Median {metric}', weight='bold', fontsize="20")
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

    plt.savefig(f'Figures/V{version}/vanilla_vs_ml/{metric}.pdf', format="pdf", bbox_inches="tight")
    plt.show()