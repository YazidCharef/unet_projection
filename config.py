import torch

NC_FILE = "/home/yazid/Documents/stage_cambridge/project_1/Pacific_Pressure_750.nc"
TYPHOON_POSITIONS_CSV = "data/typhoon_data_reordered.csv"
TYPHOON_PHASES_CSV = "data/typhoon_data_Cyclogenesis_Identification_processed.csv"

BATCH_SIZE = 4
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15

CYCLOGENESIS_RADIUS = 200
TYPHOON_RADIUS = 400
CYCLOLYSIS_RADIUS = 200

LEARNING_RATE = 0.001
NUM_EPOCHS = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEQUENCE_LENGTH = 9  
TIME_STEP = 6  