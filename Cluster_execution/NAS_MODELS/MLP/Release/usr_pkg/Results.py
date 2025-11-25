import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def load_and_extract_data(csv_file_path):
    """
    Load CSV file and extract relevant columns
    """
    try:
        df = pd.read_csv(csv_file_path)
        
        # Extract required columns
        extracted_data = df[['node_size', 'model_type', 'rmse', 'mape', 'training_time']].copy()
        
        print(f"Data loaded from {csv_file_path}:")
        print(f"Shape: {extracted_data.shape}")
        print(f"Model types found: {extracted_data['model_type'].unique()}")
        print(f"Node sizes found: {sorted(extracted_data['node_size'].unique())}")
        
        return extracted_data
    
    except FileNotFoundError:
        print(f"Error: File '{csv_file_path}' not found.")
        return None
    except KeyError as e:
        print(f"Error: Column {e} not found in the CSV file.")
        return None

def create_circuit_vs_qvalues_comparison(data, method_name):
    """
    Create comparison plots specifically for circuit level vs q_values vs Q_values_full_model
    """
    plt.style.use('default')
    sns.set_palette("Set1")
    
    # Filter for the specific model types we want to compare
    model_types_to_compare = ['Circuit', 'Q_values', 'Q_values_full_model']
    
    # Check which model types are actually present in the data
    available_models = [model for model in model_types_to_compare if model in data['model_type'].unique()]
    
    if len(available_models) < 2:
        print(f"Warning: Only {len(available_models)} of the expected model types found in {method_name} data.")
        print(f"Available models: {available_models}")
        return
    
    # Create the comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{method_name}: Circuit Level vs Q-Values vs Q-Values Full Model Comparison', 
                 fontsize=16, fontweight='bold')
    
    metrics = [('rmse', 'RMSE'), ('mape', 'MAPE (%)'), ('training_time', 'Training Time (s)')]
    
    # Plot 1: RMSE comparison
    ax1 = axes[0, 0]
    for model in available_models:
        model_data = data[data['model_type'] == model]
        
        # Regular plotting for all models, including Q_values_full_model
        grouped = model_data.groupby('node_size')['rmse'].agg(['mean', 'std']).reset_index()
        # Filter out 'full' node_size if present
        grouped = grouped[grouped['node_size'] != 'full']
        # Ensure node_size is numeric for proper plotting
        grouped = grouped[pd.to_numeric(grouped['node_size'], errors='coerce').notna()]
        
        if len(grouped) > 0:
            ax1.errorbar(grouped['node_size'], grouped['mean'], 
                        yerr=grouped['std'], marker='o', linewidth=2,
                        label=f'{model}', capsize=5, capthick=2, markersize=8)
    
    ax1.set_xlabel('Node Size')
    ax1.set_ylabel('RMSE')
    ax1.set_title('RMSE Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: MAPE comparison
    ax2 = axes[0, 1]
    for model in available_models:
        model_data = data[data['model_type'] == model]
        
        grouped = model_data.groupby('node_size')['mape'].agg(['mean', 'std']).reset_index()
        grouped = grouped[grouped['node_size'] != 'full']
        grouped = grouped[pd.to_numeric(grouped['node_size'], errors='coerce').notna()]
        
        if len(grouped) > 0:
            ax2.errorbar(grouped['node_size'], grouped['mean'], 
                        yerr=grouped['std'], marker='s', linewidth=2,
                        label=f'{model}', capsize=5, capthick=2, markersize=8)
    
    ax2.set_xlabel('Node Size')
    ax2.set_ylabel('MAPE (%)')
    ax2.set_title('MAPE Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Training Time comparison
    ax3 = axes[1, 0]
    for model in available_models:
        model_data = data[data['model_type'] == model]
        
        grouped = model_data.groupby('node_size')['training_time'].agg(['mean', 'std']).reset_index()
        grouped = grouped[grouped['node_size'] != 'full']
        grouped = grouped[pd.to_numeric(grouped['node_size'], errors='coerce').notna()]
        
        if len(grouped) > 0:
            ax3.errorbar(grouped['node_size'], grouped['mean'], 
                        yerr=grouped['std'], marker='^', linewidth=2,
                        label=f'{model}', capsize=5, capthick=2, markersize=8)
    
    ax3.set_xlabel('Node Size')
    ax3.set_ylabel('Training Time (s)')
    ax3.set_title('Training Time Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # Plot 4: Overall performance comparison (bar chart)
    ax4 = axes[1, 1]
    
    # Calculate average metrics for each model
    model_stats = {}
    for model in available_models:
        model_data = data[data['model_type'] == model]
        model_stats[model] = {
            'avg_rmse': model_data['rmse'].mean(),
            'avg_mape': model_data['mape'].mean(),
            'avg_time': model_data['training_time'].mean()
        }
    
    # Create grouped bar chart
    x = np.arange(len(available_models))
    width = 0.25
    
    rmse_values = [model_stats[model]['avg_rmse'] for model in available_models]
    mape_values = [model_stats[model]['avg_mape'] for model in available_models]
    time_values = [model_stats[model]['avg_time'] for model in available_models]
    
    # Normalize time values for visualization (scale to similar range as RMSE)
    time_normalized = np.array(time_values) / max(time_values) * max(rmse_values)
    
    ax4.bar(x - width, rmse_values, width, label='Avg RMSE', alpha=0.8)
    ax4.bar(x, mape_values, width, label='Avg MAPE', alpha=0.8)
    ax4.bar(x + width, time_normalized, width, label='Avg Time (normalized)', alpha=0.8)
    
    ax4.set_xlabel('Model Type')
    ax4.set_ylabel('Performance Metrics')
    ax4.set_title('Average Performance Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(available_models, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{method_name.lower()}_circuit_qvalues_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed comparison statistics
    print(f"\n" + "="*80)
    print(f"{method_name}: DETAILED MODEL COMPARISON")
    print("="*80)
    
    for model in available_models:
        model_data = data[data['model_type'] == model]
        print(f"\n{model}:")
        print("-" * 40)
        print(f"Number of experiments: {len(model_data)}")
        
        node_sizes = sorted([ns for ns in model_data['node_size'].unique() if ns != 'full'])
        print(f"Node sizes tested: {node_sizes}")
        
        print(f"RMSE - Mean: {model_data['rmse'].mean():.4f}, Min: {model_data['rmse'].min():.4f}, Max: {model_data['rmse'].max():.4f}")
        print(f"MAPE - Mean: {model_data['mape'].mean():.4f}, Min: {model_data['mape'].min():.4f}, Max: {model_data['mape'].max():.4f}")
        print(f"Training Time - Mean: {model_data['training_time'].mean():.4f}, Min: {model_data['training_time'].min():.4f}, Max: {model_data['training_time'].max():.4f}")

def create_simple_line_comparison(data, method_name):
    """
    Create a simple line comparison focusing on the three specific model types
    """
    plt.style.use('default')
    sns.set_palette("Set1")
    
    model_types_to_compare = ['Circuit', 'Q_values', 'Q_values_full_model']
    available_models = [model for model in model_types_to_compare if model in data['model_type'].unique()]
    
    if len(available_models) < 2:
        print(f"Warning: Only {len(available_models)} of the expected model types found.")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'{method_name}: Model Performance Comparison', fontsize=16, fontweight='bold')
    
    metrics = [('rmse', 'RMSE'), ('mape', 'MAPE (%)'), ('training_time', 'Training Time (s)')]
    
    for i, (metric, ylabel) in enumerate(metrics):
        ax = axes[i]
        
        for model in available_models:
            model_data = data[data['model_type'] == model]
            
            # Regular line plot for all models
            grouped = model_data.groupby('node_size')[metric].mean().reset_index()
            # Filter out non-numeric node sizes
            grouped = grouped[grouped['node_size'] != 'full']
            numeric_mask = pd.to_numeric(grouped['node_size'], errors='coerce').notna()
            grouped = grouped[numeric_mask]
            
            if len(grouped) > 0:
                ax.plot(grouped['node_size'], grouped[metric], marker='o', linewidth=3, 
                       label=f'{model}', markersize=8, alpha=0.8)
        
        ax.set_xlabel('Node Size')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{ylabel} vs Node Size')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if metric == 'training_time':
            ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(f'{method_name.lower()}_simple_line_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# Modified main function to include the new comparison
def main():
    """
    Main function to run the complete analysis including circuit vs q_values comparison
    """
    # Define your CSV files and their corresponding method names
    file_configs = [
        ('Models/MLP_Training_metrics.csv', 'MLP'),
        ('Models/XGB_Training_metrics.csv', 'XGB'),
        ('Models/Graph_Neural_Training_metrics.csv', 'GNN')
    ]
    
    all_data = []
    method_names = []
    
    print("Loading data from all files...")
    
    # Load data from all files
    for file_path, method_name in file_configs:
        if os.path.exists(file_path):
            data = load_and_extract_data(file_path)
            if data is not None:
                all_data.append(data)
                method_names.append(method_name)
                
                # Create circuit vs q_values comparison for this method  
                print(f"\nCreating circuit vs q_values comparison for {method_name}...")
                create_circuit_vs_qvalues_comparison(data, method_name)
                create_simple_line_comparison(data, method_name)
        else:
            print(f"Warning: File '{file_path}' not found. Skipping {method_name} analysis.")
    
    print(f"\nAnalysis complete! Generated comparison files:")
    for method in method_names:
        print(f"- {method.lower()}_circuit_qvalues_comparison.png")
        print(f"- {method.lower()}_simple_line_comparison.png")

if __name__ == "__main__":
    main()