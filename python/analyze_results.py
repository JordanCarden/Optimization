import os
import pandas as pd
import re

def aggregate_results_with_params():
    """
    Aggregates results from all optimization runs into a single CSV file,
    including the parameter set associated with the best SSE for each run.
    """
    results_dir = "results"
    output_file = "results_summary_with_params.csv"
    
    all_results = []
    
    try:
        all_files = os.listdir(results_dir)
    except FileNotFoundError:
        print(f"Error: The '{results_dir}' directory was not found.")
        return

    file_pattern = re.compile(
        r"^(?P<optimizer>\w+)_history_(?P<dataset>[\w_]+?)_run_(?P<seed>\d+)\.csv$"
    )

    for filename in all_files:
        match = file_pattern.match(filename)
        if not match:
            print(f"Skipping file with unexpected name: {filename}")
            continue
            
        parts = match.groupdict()
        optimizer = parts['optimizer']
        dataset = parts['dataset']
        seed = int(parts['seed'])
        
        file_path = os.path.join(results_dir, filename)
        
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                print(f"Warning: Empty file {filename}")
                min_sse = float('nan')
                best_params = "[]"
            else:
                best_run = df.loc[df['sse'].idxmin()]
                min_sse = best_run['sse']
                best_params = best_run['params']
            
            all_results.append({
                "optimizer": optimizer,
                "dataset": dataset,
                "seed": seed,
                "min_sse": min_sse,
                "params": best_params
            })
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(output_file, index=False)
    
    print(f"\nSuccessfully aggregated {len(all_results)} result files.")
    print(f"Summary with parameters saved to {output_file}")

if __name__ == "__main__":
    aggregate_results_with_params()