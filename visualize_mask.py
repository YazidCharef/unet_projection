import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import pandas as pd
from data_processing import TyphoonDataset
from config import NC_FILE, TYPHOON_POSITIONS_CSV, TYPHOON_PHASES_CSV, CYCLOGENESIS_RADIUS, TYPHOON_RADIUS, CYCLOLYSIS_RADIUS
import random
from datetime import timedelta

def plot_typhoon(ds, X, mask, time, typhoon_info):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Plot input data (wind speed)
    u = X[0].numpy()
    v = X[1].numpy()
    speed = np.sqrt(u**2 + v**2)
    
    # Determine the extent of the data
    lon_min, lon_max = ds.longitude.min().item(), ds.longitude.max().item()
    lat_min, lat_max = ds.latitude.min().item(), ds.latitude.max().item()
    extent = [lon_min, lon_max, lat_min, lat_max]
    
    # Plot wind speed as a colored contour
    c1 = ax1.contourf(ds.longitude, ds.latitude, speed, transform=ccrs.PlateCarree(), cmap='viridis', extent=extent)
    fig.colorbar(c1, ax=ax1, label='Wind Speed (m/s)')
    
    # Plot wind vectors (subsampled for clarity)
    quiver = ax1.quiver(ds.longitude[::10], ds.latitude[::10], 
                        u[::10, ::10], v[::10, ::10], 
                        transform=ccrs.PlateCarree(), scale=500, color='white', alpha=0.5)
    
    typhoon_name = typhoon_info.get('Typhoon Name', f"Unnamed Typhoon {typhoon_info['Typhoon Number']}")
    ax1.set_title(f"Wind Data - Typhoon {typhoon_name}")
    ax1.add_feature(cfeature.COASTLINE)
    ax1.add_feature(cfeature.BORDERS)
    ax1.set_extent(extent)
    
    # Plot the typhoon mask
    c2 = ax2.contourf(ds.longitude, ds.latitude, mask, transform=ccrs.PlateCarree(), cmap='RdYlBu', levels=[0, 0.5, 1], extent=extent)
    fig.colorbar(c2, ax=ax2, label='Typhoon Mask')
    
    ax2.set_title(f"Typhoon Mask - {typhoon_name}")
    ax2.add_feature(cfeature.COASTLINE)
    ax2.add_feature(cfeature.BORDERS)
    ax2.set_extent(extent)
    
    # Determine the phase of the typhoon
    if typhoon_info['Cyclogenesis Start'] <= time < typhoon_info['Cyclogenesis End']:
        phase = "Cyclogenesis"
    elif typhoon_info['Typhoon Start'] <= time < typhoon_info['Typhoon End']:
        phase = "Typhoon"
    elif typhoon_info['Cyclolysis Start'] <= time <= typhoon_info['Cyclolysis End']:
        phase = "Cyclolysis"
    else:
        phase = "Unknown"
    
    plt.suptitle(f"Typhoon {typhoon_name} - {time} - Phase: {phase}")
    plt.tight_layout()
    plt.savefig(f"typhoon_{typhoon_name}_{time.strftime('%Y%m%d%H')}.png")
    plt.close()

def get_random_typhoons(dataset, num_typhoons=5):
    num_available = len(dataset.merged_df)
    if num_available < num_typhoons:
        print(f"Seulement {num_available} typhons disponibles. Tous seront sélectionnés.")
        return dataset.merged_df
    else:
        return dataset.merged_df.sample(n=num_typhoons)

def visualize_random_typhoons(num_typhoons=5):
    ds = xr.open_dataset(NC_FILE, chunks={'time': 100})
    typhoon_positions_df = pd.read_csv(TYPHOON_POSITIONS_CSV, parse_dates=['Birth', 'Death (Latest)', 'Data Start', 'Data End'])
    typhoon_phases_df = pd.read_csv(TYPHOON_PHASES_CSV, parse_dates=['Cyclogenesis Start', 'Cyclogenesis End', 'Typhoon Start', 'Typhoon End', 'Cyclolysis Start', 'Cyclolysis End'])

    # Afficher des informations de débogage
    print(f"Nombre de lignes dans typhoon_positions_df : {len(typhoon_positions_df)}")
    print(f"Nombre de lignes dans typhoon_phases_df : {len(typhoon_phases_df)}")
    print(f"Colonnes dans typhoon_positions_df : {typhoon_positions_df.columns}")
    print(f"Colonnes dans typhoon_phases_df : {typhoon_phases_df.columns}")

    dataset = TyphoonDataset(NC_FILE, typhoon_positions_df, typhoon_phases_df)

    selected_typhoons = get_random_typhoons(dataset, num_typhoons)
    
    if selected_typhoons.empty:
        print("Aucun typhon à visualiser.")
        return

    for _, typhoon in selected_typhoons.iterrows():        # Choose a random time within the typhoon's lifetime
        start_time = typhoon['Cyclogenesis Start']
        end_time = typhoon['Cyclolysis End']
        random_time = start_time + timedelta(seconds=random.randint(0, int((end_time - start_time).total_seconds())))
        
        # Find the closest time index in the dataset
        time_index = np.abs(ds.time.values - np.datetime64(random_time)).argmin()

        X, mask = dataset[time_index]
        
        plot_typhoon(ds, X, mask, random_time, typhoon)
        print(f"Plotted typhoon {typhoon.get('Typhoon Name', typhoon['Typhoon Number'])} at time {random_time}")

if __name__ == "__main__":
    visualize_random_typhoons()