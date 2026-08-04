import pandas as pd
import concurrent.futures
import logging
from pathlib import Path

# 1. Setup Enterprise-Grade Logging
# Interviewers hate "print()" statements. Always use logging.
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s'
)

def process_station_file(file_path: Path) -> pd.DataFrame:
    """
    Worker Function: Parses and cleans a single NOAA station file.
    Designed to run in an isolated CPU process.
    """
    try:
        # SDE FLEX 1: Memory Optimization
        # Explicitly defining dtypes saves massive amounts of RAM and speeds up the C-engine parser.
        dtype_mapping = {
            'STATION': 'category', # Convert repeated station IDs to categorical integers
            'DATE': 'str', 
            'TMAX': 'float32',     # 32-bit floats are more than enough precision for temperature
            'TMIN': 'float32'
        }
        
        # SDE FLEX 2: Native C-Level Cleaning
        # NOAA notoriously uses -9999 as a missing value flag. 
        # Catching it here in `na_values` is much faster than running a vectorized .replace() later.
        df = pd.read_csv(
            file_path, 
            usecols=['STATION', 'DATE', 'TMAX', 'TMIN'],
            dtype=dtype_mapping,
            na_values=['-9999', '-9999.0', 'NaN'] 
        )
        
        # CPU-Bound Task: Vectorized date parsing
        df['DATE'] = pd.to_datetime(df['DATE'], format='%Y-%m-%d')
        df.set_index('DATE', inplace=True)
        
        # Quant Flex: Quality Control
        # Drop days where the station went completely offline.
        df.dropna(subset=['TMAX', 'TMIN'], how='all', inplace=True)
        
        # Note: NOAA GHCN data often records temperatures in tenths of degrees Celsius.
        # Uncomment the line below if your specific raw data requires scaling to standard Celsius.
        # df[['TMAX', 'TMIN']] = df[['TMAX', 'TMIN']] / 10.0 
        
        logging.info(f"Successfully processed {file_path.name} | Valid Rows: {len(df)}")
        return df
        
    except Exception as e:
        logging.error(f"Failed to process {file_path.name}: {str(e)}")
        # Return empty DF to ensure the pipeline doesn't crash from one corrupted file
        return pd.DataFrame() 

def ingest_data_parallel(data_dir: str, max_cores: int = 4) -> pd.DataFrame:
    """
    Manager Function: Orchestrates parallel processing of all station files.
    """
    files = list(Path(data_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError("No CSV files found in the target directory.")
        
    results = []
    
    # Bypass the GIL using ProcessPoolExecutor
    # min(len(files), max_cores) ensures we don't spawn useless idle processes.
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(len(files), max_cores)) as executor:
        
        # Map the worker function to the files
        future_to_file = {executor.submit(process_station_file, file): file for file in files}
        
        # as_completed yields futures as soon as they finish, regardless of the order they were submitted
        for future in concurrent.futures.as_completed(future_to_file):
            df_chunk = future.result()
            if not df_chunk.empty:
                results.append(df_chunk)
                
    # Concatenate all isolated dataframes into the final master dataset
    master_df = pd.concat(results, axis=0).sort_index()
    logging.info(f"ETL Complete. Total shape: {master_df.shape}")
    return master_df

# Example Execution Trigger:
# if __name__ == '__main__':
#     master_weather_df = ingest_data_parallel('./raw_noaa_data', max_cores=5)