import xarray as xr
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from scipy.spatial.distance import cdist
from config import NC_FILE, TYPHOON_POSITIONS_CSV, TYPHOON_PHASES_CSV, SEQUENCE_LENGTH, CYCLOGENESIS_RADIUS, TYPHOON_RADIUS, CYCLOLYSIS_RADIUS

class TyphoonDataset(Dataset):
    def __init__(self, nc_file, typhoon_positions_df, typhoon_phases_df, sequence_length=SEQUENCE_LENGTH):
        self.ds = xr.open_dataset(nc_file, chunks={'time': 100})
        self.merged_df = pd.merge(typhoon_positions_df, typhoon_phases_df, on=['Typhoon Number', 'Year'])
        self.sequence_length = sequence_length
        self.time_indices = list(range(len(self.ds.time) - sequence_length + 1))

    def __len__(self):
        return len(self.time_indices)

    def __getitem__(self, idx):
        time_index = self.time_indices[idx]
        
        u = self.ds.u.isel(time=slice(time_index, time_index + self.sequence_length)).values
        v = self.ds.v.isel(time=slice(time_index, time_index + self.sequence_length)).values

        masks = np.array([self.create_mask_for_time(time_index + i) for i in range(self.sequence_length)])

        X = torch.tensor(np.stack([u, v], axis=1), dtype=torch.float32)
        y = torch.tensor(masks, dtype=torch.float32)

        return X, y

    def create_mask_for_time(self, time_index):
        current_time = self.ds.time.isel(time=time_index).values

        mask = np.zeros((len(self.ds.latitude), len(self.ds.longitude)), dtype=bool)
        
        for _, typhoon in self.merged_df.iterrows():
            if typhoon['Cyclogenesis Start'] <= current_time <= typhoon['Cyclolysis End']:
                positions = pd.DataFrame(eval(typhoon['Positions']))
                positions['DateTime'] = pd.to_datetime(positions['DateTime'])
                closest_pos = positions.loc[(positions['DateTime'] - current_time).abs().idxmin()]
                
                if typhoon['Cyclogenesis Start'] <= current_time < typhoon['Typhoon Start']:
                    radius = CYCLOGENESIS_RADIUS
                elif typhoon['Typhoon Start'] <= current_time < typhoon['Cyclolysis Start']:
                    radius = TYPHOON_RADIUS
                else:
                    radius = CYCLOLYSIS_RADIUS

                lons, lats = np.meshgrid(self.ds.longitude, self.ds.latitude)
                coords = np.dstack((lats.ravel(), lons.ravel()))[0].astype(float)
                center = np.array([[float(closest_pos['Latitude']), float(closest_pos['Longitude'])]])
                distances = cdist(coords, center).reshape(lats.shape)
                circular_mask = distances <= (radius / 111)  
                
                mask = np.logical_or(mask, circular_mask)

        return mask

def get_data_loaders(batch_size=32, train_ratio=0.7, val_ratio=0.15, data_fraction=1.0):
    typhoon_positions_df = pd.read_csv(TYPHOON_POSITIONS_CSV, parse_dates=['Birth', 'Death (Latest)', 'Data Start', 'Data End'])
    typhoon_phases_df = pd.read_csv(TYPHOON_PHASES_CSV, parse_dates=['Cyclogenesis Start', 'Cyclogenesis End', 'Typhoon Start', 'Typhoon End', 'Cyclolysis Start', 'Cyclolysis End'])
    
    full_dataset = TyphoonDataset(NC_FILE, typhoon_positions_df, typhoon_phases_df)
    
    if data_fraction < 1.0:
        num_samples = int(len(full_dataset) * data_fraction)
        indices = torch.randperm(len(full_dataset))[:num_samples]
        dataset = Subset(full_dataset, indices)
    else:
        dataset = full_dataset
    
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader