import torch
import torch.nn as nn

BATCH = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEM_MAX = 4E9  # 4E9 bytes = 4GB
DEFAULT_PATH = "C:/Users/domin/OneDrive/Stalinis kompiuteris/py/audio/sensor_data/"

target_path = "src/data/sensor_data/measurements/target.csv"
manual_paths = [
    "src/data/sensor_data/measurements/raw/test_1_gp.csv",
    "src/data/sensor_data/measurements/raw/test_2_gp.csv",
    "src/data/sensor_data/measurements/raw/test_3_gp.csv",       
    "src/data/sensor_data/measurements/raw/test_4_gp.csv",
    "src/data/sensor_data/measurements/raw/test_5_gp.csv",
    "src/data/sensor_data/measurements/raw/test_6_gp.csv",
    "src/data/sensor_data/measurements/raw/test_7_gp.csv",
    "src/data/sensor_data/measurements/raw/test_8_gp.csv",
    "src/data/sensor_data/measurements/raw/test_9_gp.csv",
    "src/data/sensor_data/measurements/raw/test_10_gp.csv",
    "src/data/sensor_data/measurements/raw/test_11_gp.csv",
    "src/data/sensor_data/measurements/raw/test_12_gp.csv",
    "src/data/sensor_data/measurements/raw/test_13_gp.csv",
    "src/data/sensor_data/measurements/raw/test_26_gp.csv",
    "src/data/sensor_data/measurements/raw/test_27_gp.csv",
    "src/data/sensor_data/measurements/raw/test_28_gp.csv",
    "src/data/sensor_data/measurements/raw/test_29_gp.csv",
    "src/data/sensor_data/measurements/raw/test_30_gp.csv",
    "src/data/sensor_data/measurements/raw/test_31_gp.csv",
    "src/data/sensor_data/measurements/raw/test_32_gp.csv",
]

indices = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 26, 27, 28, 29, 30, 31, 32
]