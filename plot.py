import re
import os
import pandas as pd
import matplotlib.pyplot as plt
import glob

def parse_pagerank_output(file_path, implementation):
    """Parses a PageRank benchmark output file and extracts necessary details."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    time_match = re.search(r'Time taken to converge: ([\d\.]+) seconds', content)
    threads_match = re.search(r'PageRank will run with (\d+) threads', content)
    
    time_taken = float(time_match.group(1)) if time_match else None
    threads = int(threads_match.group(1)) if threads_match else 1  # Default to 1 for serial execution
    
    return {
        'Implementation': implementation,
        'Threads': threads,
        'Time': time_taken
    }

def analyze_pagerank_results(folders, labels):
    """Analyzes all PageRank benchmark files from multiple folders."""
    all_data = []
    
    for folder, label in zip(folders, labels):
        files = glob.glob(os.path.join(folder, "*.txt"))
        data = [parse_pagerank_output(file, label) for file in files if parse_pagerank_output(file, label)['Time'] is not None]
        all_data.extend(data)
    
    df = pd.DataFrame(all_data)
    
    if df.empty:
        print("No valid benchmark results found in the folders.")
        return
    
    # Get serial execution times per implementation
    serial_times = df[df['Threads'] == 1].set_index('Implementation')['Time'].to_dict()
    
    df['Speedup'] = df.apply(lambda row: serial_times.get(row['Implementation'], 1) / row['Time'], axis=1)
    df['Efficiency'] = df['Speedup'] / df['Threads']
    
    # Keep only one row for serial implementation
    serial_rows = df[df['Threads'] == 1].copy()
    serial_rows['Implementation'] = 'Serial Implementation'
    serial_rows = serial_rows[['Implementation', 'Threads', 'Time', 'Speedup', 'Efficiency']]
    
    # Remove duplicate serial entries from main df and retain only parallel runs
    df = df[df['Threads'] > 1]
    
    # Append the single serial implementation row back
    df = pd.concat([serial_rows, df], ignore_index=True)
    
    # Sort by implementation and thread count
    df = df.sort_values(by=['Implementation', 'Threads'])
    
    # Generate LaTeX table
    latex_code = df.to_latex(index=False, caption="PageRank Benchmark Results", label="tab:pagerank_results")
    
    print("\nGenerated LaTeX Table:\n")
    print(latex_code)
    
    # Plot speedup comparison
    plt.figure()
    for label in labels:
        subset = df[df['Implementation'] == label]
        plt.plot([1] + subset['Threads'].tolist(), [1] + subset['Speedup'].tolist(), marker='o', linestyle='-', label=f'{label} Speedup')
    
    ideal_x = sorted(df['Threads'].unique())
    ideal_y = ideal_x  # Ideal speedup is y = x
    plt.plot([1] + ideal_x, [1] + ideal_y, linestyle='--', color='green', label='Ideal Speedup')
    
    plt.xscale('log', base=2)
    plt.xticks([1] + ideal_x, [1] + ideal_x)
    plt.xlim(left=1)
    plt.ylim(bottom=1)
    plt.xlabel('Number of Threads')
    plt.ylabel('Speedup')
    plt.legend()
    plt.title('OpenMP Speedup Comparison (Log scale)')
    plt.grid(True)
    plt.show()
    
    # Plot efficiency comparison
    plt.figure()
    for label in labels:
        subset = df[df['Implementation'] == label]
        plt.plot([1] + subset['Threads'].tolist(), [1] + subset['Efficiency'].tolist(), marker='o', linestyle='-', label=f'{label} Efficiency')
    
    plt.axhline(y=1, color='r', linestyle='--', label='Ideal Efficiency')
    plt.xscale('log', base=2)
    plt.xticks([1] + ideal_x, [1] + ideal_x)
    plt.xlim(left=1)
    plt.ylim(bottom=0, top=1.2)  # Keeping efficiency bound between 0 and slightly above 1
    plt.xlabel('Number of Threads')
    plt.ylabel('Efficiency')
    plt.legend()
    plt.title('OpenMP Efficiency Comparison (Log scale)')
    plt.grid(True)
    plt.show()
    
    return latex_code

analyze_pagerank_results(['outputs/multi_optimized','outputs/multi'],['Local accumulation', 'Baseline version'])