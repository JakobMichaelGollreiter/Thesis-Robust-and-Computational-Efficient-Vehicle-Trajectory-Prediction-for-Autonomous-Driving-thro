import os
import random
from pathlib import Path


class ScenarioLoader:
    def __init__(self) -> None:
        pass

    def get_random_training_scenario(self):
        # later on this will be part of the training loop

        # try to catch random file from SUMO scenarios:
        #SCENARIO_PATH = Path("scenarios/downloadedSceanrios/commonroad-scenarios-2020a_scenarios/scenarios/recorded/SUMO")
        SCENARIO_PATH = Path("scenarios/FlensburgTrain")
        # SCENARIO_PATH = Path("scenarios/downloadedSceanrios/commonroad-scenarios-2020a_scenarios/scenarios/recorded/hand-crafted")
        # SCENARIO_PATH = Path("scenarios/downloadedSceanrios/commonroad-scenarios-2020a_scenarios/scenarios/recorded/NGSIM/Lankershim") # trafic lights really too big displayed
        # SCENARIO_PATH = Path("scenarios/downloadedSceanrios/commonroad-scenarios-2020a_scenarios/scenarios/recorded/NGSIM/Peachtree") # trafic lights really too big displayed
        # SCENARIO_PATH = Path("scenarios/downloadedSceanrios/commonroad-scenarios-2020a_scenarios/scenarios/recorded/NGSIM/US101")
        # SCENARIO_PATH = Path("scenarios/downloadedSceanrios/commonroad-scenarios-2020a_scenarios/scenarios/recorded/scenario-factory") # strage scenarios

        # Get a list of all files in the directory
        file_list = os.listdir(str(SCENARIO_PATH))

        # Filter out any non-files (e.g. directories)
        file_list = [f for f in file_list if os.path.isfile(
            os.path.join(str(SCENARIO_PATH), f))]

        # Pick a random file from the list
        random_file = random.choice(file_list)
        SCENARIO_PATH = SCENARIO_PATH.joinpath(random_file)
        return SCENARIO_PATH

    def get_specific_scenario(self):
        # use this method if you want to load a specific scenario
        # SCENARIO_PATH = 'scenarios/ZAM-Ramp-1_1-T-1.xml'
        # SCENARIO_PATH = 'scenarios/ZAM_test_overtake_on_crosswalk_dynamic_pedestrian_1.xml'
        # SCENARIO_PATH = 'scenarios/ZAM_Tjunction-1_42_T-1.xml'
        # SCENARIO_PATH = 'scenarios/DEU_Flensburg-24_1_T-1.xml'
        # SCENARIO_PATH = 'scenarios/DEU_Flensburg-48_1_T-1.xml'
        # SCENARIO_PATH = 'scenarios/ZAM-Ramp-1_1-T-1.xml'
        # SCENARIO_PATH = 'scenarios/DEU_Flensburg-29_1_T-1.xml'
        # SCENARIO_PATH = 'scenarios/ARG_Carcarana-1_1_T-1.xml'
        # SCENARIO_PATH = 'scenarios/BEL_Wervik-1_2_T-1.xml'
        # SCENARIO_PATH = 'scenarios/DEU_Flensburg-18_1_T-1.xml'
        # SCENARIO_PATH = 'scenarios/BEL_Putte-8_2_T-1.xml'
        # SCENARIO_PATH = 'scenarios/DEU_Bilderstoeckchen-2_2_T-1.xml'
        # SCENARIO_PATH = 'scenarios/ITA_CarpiCentro-9_6_T-1.xml'


        '''  TEST SCENARIOS'''
        SCENARIO_PATH = 'scenarios/ComparisonTest/DEU_Flensburg-1_1_T-1.xml'
        # SCENARIO_PATH = 'scenarios/ComparisonTest/DEU_Flensburg-14_1_T-1.xml'
        # SCENARIO_PATH = 'scenarios/ComparisonTest/DEU_Flensburg-15_1_T-1.xml'
        # SCENARIO_PATH = 'scenarios/ComparisonTest/DEU_Flensburg-36_1_T-1.xml'
        # SCENARIO_PATH = 'scenarios/ComparisonTest/DEU_Lohmar-3_1_T-1.xml'
        # SCENARIO_PATH = 'scenarios/ComparisonTest/DEU_Lohmar-7_1_T-1.xml'
        # SCENARIO_PATH = 'scenarios/ComparisonTest/DEU_Lohmar-11_1_T-1.xml'
        # SCENARIO_PATH = 'scenarios/ComparisonTest/DEU_Lohmar-12_1_T-1.xml'
        # SCENARIO_PATH = 'scenarios/ComparisonTest/DEU_Lohmar-26_1_T-1.xml'
        # SCENARIO_PATH = 'scenarios/ComparisonTest/DEU_Lohmar-32_1_T-1.xml'
        # SCENARIO_PATH = 'scenarios/ComparisonTest/DEU_Reutlingen-1_1_T-1.xml'


        return SCENARIO_PATH

    def get_scenario_folder_path(self):
        # folderPath = Path(
        #     "scenarios/downloadedSceanrios/commonroad-scenarios-2020a_scenarios/scenarios/recorded/hand-crafted")
        # folderPath = Path(
        #    "scenarios/downloadedSceanrios/commonroad-scenarios-2020a_scenarios/scenarios/recorded/SUMO")
        folderPath = Path("scenarios/FlensburgTrain")
        return folderPath
