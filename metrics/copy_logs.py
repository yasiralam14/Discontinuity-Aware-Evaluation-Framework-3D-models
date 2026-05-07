import os
import shutil

def copy_logs_nested(src_root, dest_root):
    # 1. Iterate through each Model Directory in the Source Root
    for model_name in os.listdir(src_root):
        src_model_path = os.path.join(src_root, model_name)
        dest_model_path = os.path.join(dest_root, model_name)

        # Ensure we are looking at a directory and that the matching dir exists in destination
        if os.path.isdir(src_model_path) and os.path.isdir(dest_model_path):
            
            # 2. Go "one step more" inside the source model directory
            # We list everything inside src/ModelName to find that intermediate folder
            found_logs = False
            for intermediate_dir in os.listdir(src_model_path):
                intermediate_path = os.path.join(src_model_path, intermediate_dir)
                
                # Check if this is a directory
                if os.path.isdir(intermediate_path):
                    src_logs_path = os.path.join(intermediate_path, 'logs')
                    
                    # 3. Check if the 'logs' folder exists here
                    if os.path.exists(src_logs_path):
                        dest_logs_path = os.path.join(dest_model_path, 'logs')
                        
                        try:
                            # 4. Copy the logs directory
                            # dirs_exist_ok=True allows overwriting if logs already exist in dest
                            print(f"Copying: {src_logs_path} -> {dest_logs_path}")
                            shutil.copytree(src_logs_path, dest_logs_path, dirs_exist_ok=True)
                            found_logs = True
                        except Exception as e:
                            print(f"Error copying {model_name}: {e}")
            
            if not found_logs:
                print(f"Warning: No 'logs' directory found one level deep in {model_name}")

# --- Usage ---
source_directory = '/home/salam4/models'
destination_directory = '/home/salam4/renders/models'

copy_logs_nested(source_directory, destination_directory)