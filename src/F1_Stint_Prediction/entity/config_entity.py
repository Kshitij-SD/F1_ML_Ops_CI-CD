from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir : Path
    cache_dir : Path
    raw_data_dir : Path
    data_dir : Path
    