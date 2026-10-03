import os
import sys
import matplotlib.pyplot as plt
import pickle as pkl
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(parent_dir)

from burstISP.data.burst_image_dataset import BurstImageDataset
from burstISP.utils.img_util import generate_processed_image_channel3, generate_processed_image_channel4

def main():
    # Load dataset looking at the specific Rubik's cube burst
    dataset_opt = {'dataroot': '../dataset/Inference_Set/', 'num_frames': 1, 'phase': 'test'}
    dataset = BurstImageDataset(dataset_opt)
    
    # Assuming 020_0047 is index 8 (based on your input_dirs list)
    # Adjust this index if it pulls the wrong image!
    data = dataset[7] 

    lq_tensor = data['lq'][0] 
    gt_tensor = data['gt']    
    
    with open(data['meta'], "rb") as f:
        meta_data = pkl.load(f)

    # --- APPLY THE TRAINING TRANSFORM ---
    # Transpose spatially and apply the channel swap to LQ
    lq_transformed = torch.transpose(lq_tensor, 1, 2)[[0, 2, 1, 3], :, :]
    
    # Only transpose spatially for GT (since it is standard RGB)
    gt_transformed = torch.transpose(gt_tensor, 1, 2)
    # ------------------------------------

    # Process through ISP
    lq_vis = generate_processed_image_channel4(lq_transformed, meta_data, return_np=True, black_level_substracted=True)
    gt_vis = generate_processed_image_channel3(gt_transformed, meta_data, return_np=True, black_level_substracted=True)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(lq_vis)
    axes[0].set_title("Transformed LQ Frame")
    axes[0].axis("off")

    axes[1].imshow(gt_vis)
    axes[1].set_title("Transformed GT Frame")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig("transform_test.png", bbox_inches='tight', dpi=300)
    print("Saved transform_test.png")

if __name__ == '__main__':
    main()