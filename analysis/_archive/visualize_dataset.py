import os
import sys
import argparse
import matplotlib.pyplot as plt
import pickle as pkl

# Setup paths to ensure the burstISP module can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(parent_dir)

from burstISP.data.burst_image_dataset import BurstImageDataset
from burstISP.utils.img_util import generate_processed_image_channel3, generate_processed_image_channel4

def main():
    parser = argparse.ArgumentParser(description='Visualize LQ reference and GT pairs from BurstImageDataset.')
    parser.add_argument('--input_dir', type=str, default='../dataset/RealBSR_RAW_testpatch/', 
                        help='Path to the dataset directory')
    parser.add_argument('--index', type=int, default=0, 
                        help='Index of the burst folder to visualize')
    args = parser.parse_args()

    # Initialize dataset. num_frames=1 ensures we only get the reference LQ frame.
    dataset_opt = {'dataroot': args.input_dir, 'num_frames': 1, 'phase': 'test'}
    dataset = BurstImageDataset(dataset_opt)

    if args.index >= len(dataset):
        print(f"Error: Index {args.index} out of bounds. The dataset only contains {len(dataset)} items.")
        return

    # Fetch data pair
    data = dataset[args.index]

    # The dataset returns lq as [N, C, H, W]. 
    # Since num_frames=1, we take the 0th element to get the [4, H/2, W/2] tensor.
    lq_tensor = data['lq'][0] 
    # GT is loaded natively as [3, H, W]
    gt_tensor = data['gt']    
    meta_path = data['meta']

    print(f"Loading Burst Directory: {os.path.basename(data['lq_path'])}")

    # Load the corresponding metadata dictionary
    with open(meta_path, "rb") as f:
        meta_data = pkl.load(f)

    # 1. Process 4-channel LQ tensor through ISP pipeline
    try:
        lq_vis = generate_processed_image_channel4(
            lq_tensor, 
            meta_data, 
            return_np=True, 
            black_level_substracted=True
        )
    except Exception as e:
        print(f"Error applying ISP to LQ frame: {e}")
        return

    # 2. Process 3-channel GT tensor through ISP pipeline
    try:
        gt_vis = generate_processed_image_channel3(
            gt_tensor, 
            meta_data, 
            return_np=True, 
            black_level_substracted=True
        )
    except Exception as e:
        print(f"Error applying ISP to GT frame: {e}")
        return

    # 3. Plotting with Matplotlib
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    axes[0].imshow(lq_vis)
    axes[0].set_title("LQ Reference Frame (Processed 4-channel RAW)", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(gt_vis)
    axes[1].set_title("GT Frame (Processed 3-channel RAW)", fontsize=12)
    axes[1].axis("off")

    plt.tight_layout()
    
    # Create a dynamic filename based on the burst directory
    burst_name = os.path.basename(data['lq_path'])
    save_path = f"{burst_name}_comparison.png"
    
    # Save the figure to disk instead of showing it
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Visualization successfully saved to: {os.path.abspath(save_path)}")

if __name__ == '__main__':
    main()