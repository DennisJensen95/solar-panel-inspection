import json
import copy


class ConfActionStepOne:
    def __init__(self):
        super().__init__()

        self.configuration = {"Classification": None, "Model": None, "Labels": 0}

    def make_configuration(
        self,
        classification,
        model,
        learning_rate,
        momentum,
        weight_decay,
        step_size,
        gamma,
    ):
        conf = copy.copy(self.configuration)
        conf["Classification"] = classification
        conf["Model"] = model
        conf["Labels"] = self.get_number_labels(classification)
        conf["LearningRate"] = learning_rate
        conf["Momentum"] = momentum
        conf["WeightDecay"] = weight_decay
        conf["StepSize"] = step_size
        conf["Gamma"] = gamma
        return conf

    def get_number_labels(self, classification):
        data = {"binary": 2, "fault-classification": 5}
        return data[classification]

    def get_configurations(self):
        configurations = [
            self.make_configuration(
                "fault-classification",
                "faster",
                learning_rate=0.05,
                momentum=0.9,
                weight_decay=0.05,
                step_size=3,
                gamma=0.1,
            ),
            self.make_configuration(
                "fault-classification",
                "faster",
                learning_rate=0.00001,
                momentum=0.1,
                weight_decay=0.0000005,
                step_size=3,
                gamma=0.9,
            )
        ]

        return configurations

    def write_configuration_file(self, filepath, dict):
        with open(filepath, "w+") as file:
            json.dump(dict, file)
