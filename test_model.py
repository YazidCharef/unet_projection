import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from models.unet import TemporalUNet
from data_processing import TyphoonDataset, get_data_loaders
from config import DEVICE, BATCH_SIZE, NC_FILE, TYPHOON_POSITIONS_CSV, TYPHOON_PHASES_CSV, SEQUENCE_LENGTH
import random
from datetime import timedelta
import os
from matplotlib.colors import LinearSegmentedColormap

THRESHOLD = 0.2  # Threshold for binary prediction
OUTPUT_FOLDER = "typhoon_movement_visualizations"
NUM_SAMPLES_PER_CATEGORY = 50

def get_typhoon_state(typhoon, date):
    if typhoon['Cyclogenesis Start'] <= date < typhoon['Typhoon Start']:
        return 'Cyclogenesis'
    elif typhoon['Typhoon Start'] <= date < typhoon['Cyclolysis Start']:
        return 'Full Typhoon'
    elif typhoon['Cyclolysis Start'] <= date <= typhoon['Cyclolysis End']:
        return 'Cyclolysis'
    else:
        return 'Unknown'

def get_dates_by_category(dataset):
    no_typhoon_dates = []
    full_typhoon_dates = []
    cyclogenesis_dates = []
    cyclolysis_dates = []
    multiple_typhoon_dates = []

    all_dates = pd.date_range(start=dataset.ds.time.values[0], end=dataset.ds.time.values[-1])
    
    for date in all_dates:
        typhoons_on_date = dataset.merged_df[
            (dataset.merged_df['Cyclogenesis Start'] <= date) & 
            (dataset.merged_df['Cyclolysis End'] >= date)
        ]
        
        num_typhoons = len(typhoons_on_date)
        
        if num_typhoons == 0:
            no_typhoon_dates.append((date, None))
        elif num_typhoons == 1:
            typhoon = typhoons_on_date.iloc[0]
            state = get_typhoon_state(typhoon, date)
            if state == 'Full Typhoon':
                full_typhoon_dates.append((date, typhoon))
            elif state == 'Cyclogenesis':
                cyclogenesis_dates.append((date, typhoon))
            elif state == 'Cyclolysis':
                cyclolysis_dates.append((date, typhoon))
        else:
            multiple_typhoon_dates.append((date, typhoons_on_date))

    return {
        'no_typhoon': random.sample(no_typhoon_dates, min(NUM_SAMPLES_PER_CATEGORY, len(no_typhoon_dates))),
        'full_typhoon': random.sample(full_typhoon_dates, min(NUM_SAMPLES_PER_CATEGORY, len(full_typhoon_dates))),
        'cyclogenesis': random.sample(cyclogenesis_dates, min(NUM_SAMPLES_PER_CATEGORY, len(cyclogenesis_dates))),
        'cyclolysis': random.sample(cyclolysis_dates, min(NUM_SAMPLES_PER_CATEGORY, len(cyclolysis_dates))),
        'multiple_typhoons': random.sample(multiple_typhoon_dates, min(NUM_SAMPLES_PER_CATEGORY, len(multiple_typhoon_dates)))
    }

def create_custom_cmap(num_steps):
    colors = [(1, 1, 1)] + [(i/(num_steps-1), i/(num_steps-1), i/(num_steps-1)) for i in range(num_steps-1)]
    return LinearSegmentedColormap.from_list("custom_gray", colors, N=num_steps)

def visualize_results(model, dataset, dates_by_category):
    model.eval()
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    for category, dates_and_info in dates_by_category.items():
        for i, (date, typhoon_info) in enumerate(dates_and_info):
            time_index = np.abs(dataset.ds.time.values - np.datetime64(date)).argmin()
            
            inputs, targets = dataset[time_index]
            inputs = inputs.unsqueeze(0).to(DEVICE)
            targets = targets.unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                outputs = model(inputs)
                outputs = torch.sigmoid(outputs)
                
                # Create a figure with 4 subplots
                fig, axs = plt.subplots(2, 2, figsize=(20, 20))
                
                # Plot input wind map
                wind_map = torch.sqrt(inputs[0, 0, 0]**2 + inputs[0, 1, 0]**2).cpu().numpy()
                im0 = axs[0, 0].imshow(wind_map, cmap='viridis')
                axs[0, 0].set_title('Input Wind Map')
                plt.colorbar(im0, ax=axs[0, 0], label='Wind Speed')
                
                custom_cmap = create_custom_cmap(SEQUENCE_LENGTH + 1)
                
                # Plot superposed true masks
                true_mask_superposed = np.zeros_like(targets[0, 0].cpu().numpy())
                for t in range(SEQUENCE_LENGTH):
                    true_mask = targets[0, t].cpu().numpy()
                    true_mask_superposed = np.maximum(true_mask_superposed, true_mask * (t + 1) / SEQUENCE_LENGTH)
                    contour = axs[0, 1].contour(true_mask, levels=[0.5], colors=[plt.cm.viridis(t/SEQUENCE_LENGTH)], linewidths=2)
                im1 = axs[0, 1].imshow(true_mask_superposed, cmap=custom_cmap, vmin=0, vmax=1)
                axs[0, 1].set_title('True Mask (Superposed)')
                plt.colorbar(im1, ax=axs[0, 1], label='Time Step')
                
                # Plot superposed predictions
                pred_mask_superposed = np.zeros_like(outputs[0, 0, 0].cpu().numpy())
                for t in range(SEQUENCE_LENGTH):
                    pred_mask = outputs[0, 0, t].cpu().numpy()
                    pred_mask_superposed = np.maximum(pred_mask_superposed, pred_mask * (t + 1) / SEQUENCE_LENGTH)
                    contour = axs[1, 0].contour(pred_mask, levels=[0.5], colors=[plt.cm.viridis(t/SEQUENCE_LENGTH)], linewidths=2)
                im2 = axs[1, 0].imshow(pred_mask_superposed, cmap=custom_cmap, vmin=0, vmax=1)
                axs[1, 0].set_title('Prediction (Superposed)')
                plt.colorbar(im2, ax=axs[1, 0], label='Time Step')
                
                # Plot superposed binary predictions
                binary_pred_mask_superposed = np.zeros_like(outputs[0, 0, 0].cpu().numpy())
                for t in range(SEQUENCE_LENGTH):
                    binary_pred_mask = (outputs[0, 0, t].cpu().numpy() > THRESHOLD).astype(float)
                    binary_pred_mask_superposed = np.maximum(binary_pred_mask_superposed, binary_pred_mask * (t + 1) / SEQUENCE_LENGTH)
                    contour = axs[1, 1].contour(binary_pred_mask, levels=[0.5], colors=[plt.cm.viridis(t/SEQUENCE_LENGTH)], linewidths=2)
                im3 = axs[1, 1].imshow(binary_pred_mask_superposed, cmap=custom_cmap, vmin=0, vmax=1)
                axs[1, 1].set_title(f'Binary Prediction (Superposed, Threshold: {THRESHOLD})')
                plt.colorbar(im3, ax=axs[1, 1], label='Time Step')
                
                if category == 'no_typhoon':
                    typhoon_names = ["No Typhoon"]
                elif category == 'multiple_typhoons':
                    typhoon_names = [f"{t['Typhoon Name_x']} ({get_typhoon_state(t, date)})" for _, t in typhoon_info.iterrows()]
                else:
                    typhoon_names = [f"{typhoon_info['Typhoon Name_x']} ({get_typhoon_state(typhoon_info, date)})"]
                
                typhoon_info_text = "\n".join(typhoon_names)
                plt.suptitle(f'Category: {category.capitalize()}\nDate: {date}\n\nTyphoons:\n{typhoon_info_text}', fontsize=16)
                
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                
                filename = f"{category}_{date.strftime('%Y%m%d')}_movement_superposed.png"
                plt.savefig(os.path.join(OUTPUT_FOLDER, filename))
                plt.close()
                
                print(f"Saved result: {filename}")

