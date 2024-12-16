import pandas as pd

# Read the CSV file
# df = pd.read_csv('image/train/brandenburg_gate/pair_covisibility.csv')

# # Display the first few rows
# print(df.head())

# # Access specific columns or rows
# print(df['pair'])  # Access a column
# print(df.iloc[0])         # Access the first row

import csv

# Input and output file paths
# name = 'brandenburg_gate'
# name = 'grand_place_brussels'
name = 'sagrada_familia'
input_csv = f'image/train/{name}/pair_covisibility.csv'  # Replace with your CSV file path
output_txt = f'{name}.txt'  # Output text file to save the names

# Open the CSV and process the first column
with open(input_csv, mode='r') as csv_file:
    reader = csv.reader(csv_file)
    
    # Open the output text file for writing
    with open(output_txt, mode='w') as txt_file:
        for i, row in enumerate(reader):
            if i == 0:
                continue  # Skip the header row if present
            # Extract the first column
            first_column = row[0]
            # Split the names by '_'
            image_names = first_column.split('-')
            # Write each image name to the text file
            # print(f"image/train/{name}/images/{image_names[0]}.jpg image/train/{name}/images/{image_names[0]}.jpg")
            txt_file.write(f"image/train/{name}/images/{image_names[0]}.jpg image/train/{name}/images/{image_names[1]}.jpg\n")

print(f"Image names have been written to {output_txt}.")
