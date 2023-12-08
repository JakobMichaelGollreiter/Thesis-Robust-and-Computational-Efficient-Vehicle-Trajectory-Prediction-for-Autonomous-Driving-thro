import csv
import numpy as np
import datetime
import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from numpy import asarray


class DataGenerator:
    def __init__(self) -> None:
        pass

    def make_input_and_output_csv_files(self, comprehensiveTrainingData, SCENARIO_PATH):
        """
        Save the given comprehensive training data to a CSV file.

        Args:
        - comprehensiveTrainingData (dict): A dictionary containing comprehensive training data.
        - SCENARIO_PATH (str): The path of the scenario file.

        Returns:
        - file_path (str): The path of the saved CSV file.
        """

        filename = os.path.splitext(os.path.basename(SCENARIO_PATH))[0]  # Specify the file path
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y_%m_%d_%H:%M:%S")
        file_path = f'db39/big/{filename}_{timestamp}TrainingData.csv'

        with open(file_path, 'w', newline='') as csvfile:  # Open the CSV file in write mode
            writer = csv.writer(csvfile)
            writer.writerow(['filename', 'cr_id', 'time_begin', 'egoVelocity', 'mapInView',
                            'socialInformationOfCarsInView', 'groundTruthTrajectory'])  # Write the header row
            for cr_id, cr_id_data in comprehensiveTrainingData.items(): # Iterate over the dictionary entries and write each row
                for time_begin, data in cr_id_data.items():
                    egoVelocity = data['egoVelocity']
                    # with no conversion to list or tensor
                    mapInView = data['mapInView']
                    groundTruthTrajectory = data['groundTruthTrajectory']
                    socialInformationOfCarsInView = data['socialInformationOfCarsInView']

                    # Convert numpy arrays to lists before writing to CSV:
                    mapInView_list = mapInView.tolist()
                    groundTruthTrajectory_list = groundTruthTrajectory.tolist()
                    socialInformationOfCarsInView_list = socialInformationOfCarsInView.tolist()
                    writer.writerow([filename, cr_id, time_begin, egoVelocity, mapInView_list,
                                    socialInformationOfCarsInView_list, groundTruthTrajectory_list])
                    # writer.writerow([cr_id, time_begin, mergedArray, relativePositionAndVelocityToEgo, outputVelocitiesAtSpecificTime])
        print(f"CSV file saved successfully at: {file_path}")
