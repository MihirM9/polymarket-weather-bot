import pandas as pd
import concurrent.futures
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s'
)


def process_station_file(file_path: Path) -> pd.DataFrame:
    try:
        dtype_mapping = {
            'STATION': 'category',
            'DATE': 'str',
            'TMAX': 'float32',
            'TMIN': 'float32'
        }

        df = pd.read_csv(
            file_path,
            usecols=['STATION', 'DATE', 'TMAX', 'TMIN'],
            dtype=dtype_mapping,
            na_values=['-9999', '-9999.0', 'NaN']
        )

        df['DATE'] = pd.to_datetime(df['DATE'], format='%Y-%m-%d')
        df.set_index('DATE', inplace=True)

        # Remove missing observations
        df.dropna(subset=['TMAX', 'TMIN'], how='all', inplace=True)

        logging.info(
            f"Successfully processed {file_path.name} | Valid Rows: {len(df)}"
        )
        return df

    except Exception as e:
        logging.error(f"Failed to process {file_path.name}: {str(e)}")
        return pd.DataFrame()


def ingest_data_parallel(
    data_dir: str,
    max_cores: int = 4
) -> pd.DataFrame:

    files = list(Path(data_dir).glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            "No CSV files found in the target directory."
        )

    results = []

    # Parallel ETL
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=min(len(files), max_cores)
    ) as executor:

        future_to_file = {
            executor.submit(process_station_file, file): file
            for file in files
        }

        for future in concurrent.futures.as_completed(future_to_file):

            df_chunk = future.result()

            if not df_chunk.empty:
                results.append(df_chunk)

    master_df = pd.concat(results, axis=0).sort_index()

    logging.info(
        f"ETL Complete. Total shape: {master_df.shape}"
    )

    return master_df
