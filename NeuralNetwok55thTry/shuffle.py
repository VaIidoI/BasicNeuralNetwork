import random

def shuffle_file_lines(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    random.shuffle(lines)
    
    with open(file_path, 'w') as file:
        file.writelines(lines)

# Example usage
file_path = 'train.csv'  # Update this to the path of your file
shuffle_file_lines(file_path)
