import os
import re
import yaml



def access_specific_value_in_log_files(base_folder, target_file_name, line_number):
    folder_list = sorted(os.listdir(base_folder),reverse=True)


    for folder_name in folder_list:
        folder_path = os.path.join(base_folder, folder_name)
        if os.path.isdir(folder_path):
            target_file_path = os.path.join(folder_path, target_file_name)
            hydrapath=folder_path+"/.hydra"+"/config.yaml"

            with open(hydrapath, 'r') as file:
                config = yaml.safe_load(file)
            if os.path.isfile(target_file_path):
                with open(target_file_path, 'r') as file:
                    lines = file.readlines()
                    if len(lines) >= line_number:
                        line = lines[line_number - 1]  # line_number is 1-based
                        # Extract the value using regex
                        match = re.search(r'\(\d+, ([\d\.]+)\)', line)
                        if match:
                            value = match.group(1)
                            maxattackval=int(config['max_attack_ratio']*10)
                            labelval=int(config['label_attack_ratio']*100)

                            print(f"({maxattackval}, {labelval}, {value})")
                            
                        else:
                            print(f"No match found in line {line_number} of {target_file_path}")
                    else:
                        print(f"{target_file_path} does not have {line_number} lines")
            else:
                print(f"{target_file_name} does not exist in {folder_path}")

# Example usage
base_folder = '/Users/charansr/Stamp2024/3d100_outputs/_2024-06-21'
target_file_name = 'main.log'
line_number = 197
access_specific_value_in_log_files(base_folder, target_file_name, line_number)
