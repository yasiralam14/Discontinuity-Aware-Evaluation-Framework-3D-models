import pandas as pd
import numpy as np


tnt_scenes = [
        'Barn', 'Caterpillar', 'Courthouse', 'Ignatius', 'Meetingroom', 'Truck'
]
mp360_scenes = ['bicycle', 'flowers', 'garden', 'stump', 'treehill', 'bonsai', 'counter', 'kitchen', 'room']

def assign_dataset(scene_name):
    if scene_name in tnt_scenes:
        return 'tnt'
    elif scene_name in mp360_scenes:
        return 'mp360'
    else:
        return 'unknown' 


df = pd.read_csv('/home/salam4/renders/models/consolidated_results.csv')
df = df[df['Scene'] != 'average']

df['Dataset'] = df['Scene'].apply(assign_dataset)


std_cols = ['Model', 'Dataset', 'Scene', 'LPIPS', 'PSNR', 'SSIM']
df_std = df[std_cols].melt(
    id_vars=['Model', 'Dataset', 'Scene'], 
    var_name='Metric', 
    value_name='Standard Value'
)

# 5. MELT: Reshape Masked Metrics (Masked LPIPS, Masked PSNR, Masked SSIM)
masked_cols = ['Model', 'Dataset', 'Scene', 'Masked LPIPS', 'Masked PSNR', 'Masked SSIM']
df_masked = df[masked_cols].melt(
    id_vars=['Model', 'Dataset', 'Scene'], 
    var_name='Metric', 
    value_name='Masked Value'
)


df_masked['Metric'] = df_masked['Metric'].str.replace('Masked ', '')

# 6. MERGE: Combine them side-by-side
final_df = pd.merge(df_std, df_masked, on=['Model', 'Dataset', 'Scene', 'Metric'])

# 7. CALCULATE: Create the Difference column
# calculating Standard - Masked
final_df['Difference'] = np.where(
    final_df['Metric'] == 'LPIPS', 
    final_df['Masked Value'] - final_df['Standard Value'], 
    final_df['Standard Value'] - final_df['Masked Value']
)

# 8. EXPORT: Reorder columns and save
output_cols = ['Model', 'Dataset', 'Scene', 'Metric', 'Standard Value', 'Masked Value', 'Difference']
final_df = final_df[output_cols]

final_df.to_csv('/home/salam4/renders/models/reshaped_output.csv', index=False)

print("Conversion complete. Saved to 'reshaped_output.csv'")