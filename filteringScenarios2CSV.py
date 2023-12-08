
import os
import csv
import glob
import pandas as pd
from tqdm import tqdm  # Import the tqdm library for the progress bar
import torch
import random
import numpy as np
# import HelperFunctions as HelperFunctions
from commonroad_prediction import HelperFunctions
from commonroad_prediction import toPointListConversion

'''
MODULE FUNCTIONALITY:
    1) Combining CSV Files: 
        - The module first navigates to a specified directory and identifies all the CSV files present in that directory.
        - All CSV filenames are stored in a list called valid_filenames. 
        - The data from all the valid CSV files is then combined into a single DataFrame, and the combined DataFrame is exported to a CSV file named "combined.csv".
    2) Filtering Bad Data: 
        - The module proceeds to read the "combined.csv" file and filters out bad data points based on certain criteria. 
        - Each row in the CSV file is processed as a dictionary, and specific fields are extracted and converted to appropriate data types. 
        - The filtering process is performed based on conditions related to the data fields, such as checking for corrupt entries, abnormal values, and specific patterns in the trajectory data.
    3) Saving Filtered Data:
        - The filtered data is stored in a new CSV file named "cleanedCombine.csv" in the same directory.
        - The fields in the filtered data include the filename,  cr_id, time_begin, egoVelocity, mapInView, socialInformationOfCarsInView, and groundTruthTrajectory.
'''

os.chdir("db39/trainPredictionLocal")
dir = os.getcwd()

# 1) Combining CSV Files: 
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

valid_filenames = [] # Create an empty list to store valid CSV filenames

for filename in tqdm(all_filenames, desc = "making prelimary check"): # Iterate through each CSV file
    df = pd.read_csv(filename) # Read the CSV file
    valid_filenames.append(filename)

combined_csv = pd.concat([pd.read_csv(f) for f in valid_filenames])# Combine all valid files in the list
combined_csv.to_csv("combined.csv", index=False, encoding='utf-8-sig')# Export to CSV
print(f'loading combined.csv...')

# 2) Filtering Bad Data: 
csv_file = os.path.join(dir,'combined.csv')
new_csv_file = os.path.join(dir,'cleanedCombine.csv')  # Path for the new CSV file
data_list = []  # List to store the data as dictionaries
with open(csv_file, 'r') as file: # Read the CSV file and store data as dictionaries in the data_list
    reader = csv.DictReader(file)
    for row in reader:
        data_list.append(row)

print(f'datalist first entry = {data_list[0]}')        
filtered_data_list = [] # Create a list to store the filtered data
threshold = 0.1
maxEntry = 0 # maxEntry to show the max value of any entry in the dataset that is filtered out

for entry in tqdm(data_list, desc="Filtering data"): 
    try: # try loading dataPoint. If fails(e.g. nan entry) skip corrupted data
        filename = entry['\ufefffilename']
        cr_id = int(entry['cr_id'])
        time_begin =int(entry['time_begin'])
        ego_velocity = float(entry['egoVelocity'])
        map_in_view = eval(entry['mapInView'])
        social_info_of_cars = eval(entry['socialInformationOfCarsInView'])
        ground_truth_trajectory = eval(entry['groundTruthTrajectory'])
    except:
        continue   

    badEntry = False
    if ego_velocity == 0:
        randomNum = random.uniform(0, 1)
        if randomNum < 0.8:
            continue

    modGroundTruthTrajectory = torch.tensor([[entry[0], entry[1]] for entry in ground_truth_trajectory]).view(30,2) # ignore the last entry of the ground truth trajectory as this is just a number indicating how the trajectory was generated 
    modGroundTruthTrajectoryYValues = torch.tensor([[entry[1]] for entry in modGroundTruthTrajectory]).view(30,1) # ignore the last entry of the ground truth trajectory as this is just a number indicating how the trajectory was generated 
    modGroundTruthTrajectoryXValues = torch.tensor([[entry[0]] for entry in modGroundTruthTrajectory]).view(30,1) # ignore the last entry of the ground truth trajectory as this is just a number indicating how the trajectory was generated 
  
    if (torch.abs(modGroundTruthTrajectoryYValues) > 0.7).any() or (torch.abs(modGroundTruthTrajectoryXValues) > 7).any() or (torch.abs(torch.tensor(ego_velocity)) > 20).any(): # check for bad data and if so skip it
        continue
    
    onroad = toPointListConversion.outputTrajectoryOnDrivableRoadScalar(modGroundTruthTrajectory, map_in_view)
    if onroad < 0.85:
        print(f'not onroad {filename} {cr_id} {time_begin}')
        continue


    if ego_velocity > maxEntry: maxEntry = ego_velocity
    for i in range(20):
        x = social_info_of_cars[i][0]
        y = social_info_of_cars[i][1]
        dx = social_info_of_cars[i][2]
        dy = social_info_of_cars[i][3]
        if x > maxEntry: maxEntry = x
        if y > maxEntry: maxEntry = y
        if dx > maxEntry: maxEntry = dx
        if dy > maxEntry: maxEntry = dy
    for i in range(29):
        groundTruthX = ground_truth_trajectory[i][0]
        groundTruthY = ground_truth_trajectory[i][1]
        if groundTruthX > maxEntry: maxEntry = groundTruthX
        if groundTruthY > maxEntry: maxEntry = groundTruthY
        if groundTruthX == 0 and groundTruthY == 0 and ground_truth_trajectory[i+1][0] > 0.03 and ground_truth_trajectory[i+1][1] > 0.003: # if the car is not moving but the next entry is moving rapidely, then it is a corrupted entry
            badEntry = True
            print(f'{filename} {cr_id} {time_begin}')
            break
        if(abs(groundTruthY)/abs(groundTruthX+1) > 0.1):
            badEntry = True
            break
        
    if badEntry:
        continue

    filtered_data_list.append({
        '\ufefffilename': filename,
        'cr_id': cr_id,
        'time_begin': time_begin,
        'egoVelocity': ego_velocity,
        'mapInView': str(map_in_view),
        'socialInformationOfCarsInView': str(social_info_of_cars),
        'groundTruthTrajectory': str(ground_truth_trajectory)
    })
# 3) Saving Filtered Data:
with open(new_csv_file, 'w', newline='') as new_file: # Write the filtered data to the new CSV file
    fieldnames = ['\ufefffilename', 'cr_id', 'time_begin', 'egoVelocity', 'mapInView',
                  'socialInformationOfCarsInView', 'groundTruthTrajectory']
    writer = csv.DictWriter(new_file, fieldnames=fieldnames)
    writer.writeheader()# Write the header row
    for entry in tqdm(filtered_data_list, desc="Writing cleaned data"):  # Add the progress bar here
        writer.writerow(entry)

print(f"cleaned CSV file saved successfully at: {new_csv_file}")
# print(f'the biggest filtered data entry was = {maxEntry}')