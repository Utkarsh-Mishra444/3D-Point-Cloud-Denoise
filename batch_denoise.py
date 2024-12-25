import os
import subprocess
import argparse

def batch_denoise(data_path, model_path, save_path):
    # Get list of all .xyz files in the data_path
    xyz_files = [f for f in os.listdir(data_path) if f.endswith('.xyz')]

    if not xyz_files:
        print("No .xyz files found in the specified data path.")
        return

    os.makedirs(save_path, exist_ok=True)  # Create save_path if it doesn't exist

    for xyz_file in xyz_files:
        input_file = os.path.join(data_path, xyz_file)
        output_file = os.path.join(save_path, f"denoised_{xyz_file}")
        
        # Command to execute
        command = [
            "python", "denoise_object.py",
            "--data_path", input_file,
            "--save_path", output_file,
            "--model_path", model_path
        ]
        
        print(f"Processing {xyz_file}...")
        
        # Run the command
        subprocess.run(command)
    
    print("All files processed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process .xyz files for denoising.")
    parser.add_argument("--data_path", required=True, help="Path to the directory containing .xyz files.")
    parser.add_argument("--model_path", required=True, help="Path to the trained model.")
    parser.add_argument("--save_path", required=True, help="Path to the directory to save denoised outputs.")
    
    args = parser.parse_args()

    batch_denoise(args.data_path, args.model_path, args.save_path)
