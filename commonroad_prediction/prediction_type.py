# globals
PREDICTION_TYPES = ["Linear", "CYRA", "Ground_truth", "Lanebased"]


class PredictionType:

    def __init__(self, prediction_type: str):

        if prediction_type not in PREDICTION_TYPES:
            self.type = "Ground_truth"
        else:
            self.type = prediction_type
