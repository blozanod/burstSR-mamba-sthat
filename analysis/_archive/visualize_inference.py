import os
import sys
import argparse
import yaml
import copy
import torch
import cv2
import random
import numpy as np
import pickle as pkl

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(parent_dir)

from burstISP.archs.mambafusion_arch import MambaFusionNet
from burstISP.utils.img_util import img2tensor, imwrite, imfrombytes
from burstISP.utils.img_util import generate_processed_image_channel3 
from burstISP.data.burst_image_dataset import BurstImageDataset

def main():
    parser = argparse.ArgumentParser(description='Run inference and visualize results using ISP pipeline.')
    parser.add_argument('--config', type=str, default='../experiments/MambaFusion_x4/config.yml')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--input_dir', type=str, default='../dataset/RealBSR_RAW_testpatch/')
    parser.add_argument('--output_path', type=str, default='./inferences_vis/')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        opt = yaml.safe_load(f)
    
    network_opt = opt.get('network_g', {})
    num_frames = network_opt.get('num_frames', 5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Initializing MambaFusionNet on {device}...")
    model = MambaFusionNet(**network_opt)
    
    print(f"Loading weights from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    state_dict = checkpoint.get('params_ema', checkpoint.get('params', checkpoint.get('state_dict', checkpoint)))
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    input_dirs = ["010_0023", "010_0104", "013_0265", "010_0292", "020_0543",
                  "014_0674", "006_0291", "007_0065", "020_0047", "027_0388"]
                  
    os.makedirs(args.output_path, exist_ok=True)

    # Natively use Dataset to guarantee training/inference parity
    dataset_opt = {'dataroot': args.input_dir, 'num_frames': num_frames, 'phase': 'test'}
    dataset = BurstImageDataset(dataset_opt)
    
    # Ensure fixed sampling matches visualizer intent
    random.seed(42)

    for i in range(len(dataset)):
        data = dataset[i]
        burst_dir = data['lq_path']
        directory = os.path.basename(burst_dir)
        
        if directory not in input_dirs:
            continue
            
        print(f"\n---> Processing burst: {directory}")
        
        with open(data['meta'], "rb") as f:
            meta_data = pkl.load(f)

        input_tensor = data['lq'].unsqueeze(0).to(device)
        
        print(f"Running inference. Input shape: {input_tensor.shape}")
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                output_tensor = model(input_tensor)
            
        if isinstance(output_tensor, tuple):
            output_tensor = output_tensor[0]
        output_tensor = output_tensor.squeeze(0).float() 

        try:
            vis_img = generate_processed_image_channel3(
                output_tensor, 
                copy.deepcopy(meta_data), 
                return_np=True, 
                black_level_substracted=True 
            )
        except Exception as e:
            print(f"Error during ISP processing: {e}")
            continue
        
        save_filename = os.path.join(args.output_path, f"{directory}_vis.png")
        
        # FIX: The ISP yields a true [R, G, B] array. OpenCV's imwrite safely needs it converted
        # back to BGR to avoid writing out physically swapped PNG files. 
        vis_img_bgr = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
        imwrite(vis_img_bgr, save_filename)
        print(f"Output visually saved to: {os.path.abspath(save_filename)}")

if __name__ == '__main__':
    main()