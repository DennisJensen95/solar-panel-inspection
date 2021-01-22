import numpy as np
import torchvision
import torch
import copy
import sys      

class LogHelpers:

    def __init__(self, binary=False):
        self.img = []
        self.tar = []
        self.binary = binary

        self.label_data_struct = np.array([0, 0, 0, 0, 0, 0, np.array([])])
        
    def __load__(self, target, image):
        self.tar = target
        self.img = image

    def _get_existing_labels(self):
        fault = LabelEncoder(binary=self.binary)
        return fault.fault_key_to_value
        
    def _get_highest_prediction(self):
        labels = self._get_existing_labels()
       
        if len(self.img["scores"] > 0):
            score = self.img["scores"][0].numpy()
            label = labels.get(int(self.img["labels"][0].numpy()), "Out of bounds")

            return label, score

        return "No Prediction", 0

    def get_highest_predictions(self, score_limit=0.90):
        labels = self._get_existing_labels()

        score = []
        label = []

        # Check if there are any predictions
        # and if the best prediction is good enough
        if len(self.img["scores"]) > 0 and self.img["scores"][0] >= score_limit:
            # iterate through predictions
            for i in range(len(np.array(self.img["scores"].numpy()))):
                # if score is good enough, we keep it
                if self.img["scores"][i] >= score_limit:
                    label.append(
                        labels.get(int(self.img["labels"][i].numpy()), "Out of bounds")
                    )
                    score.append(float(self.img["scores"][i].numpy()))
                else:  # at this point, all other scores are lower, so we break the loop
                    break

            # return what we obtained
            return label, score

        # if no score is above the threshold, we return the highest score
        elif len(self.img["scores"]) > 0 and self.img["scores"][0] < score_limit:
            # return self._get_highest_prediction()
            return "No Prediction", 0

        else:
            # if no prediction is made, we return default
            return "No Prediction", 0

    def add_stat_to_data(self, data, label, result):
        if not isinstance(label, int):
            label = int(label)
        add = [result=="Tpos", result=="Tneg", result=="Fpos", result=="Fneg", 0, 0]
        data[label][:-1] += add
        
        return data
    
    def add_wrong_pred_label_to_data(self, data, label, result):
        if not isinstance(label, int):
                label = int(label)
        fp_array = data[label][-1]
        new_fp_array = np.append(fp_array, np.array(result))
        data[label][-1] = new_fp_array
        
        return data
    
    def add_total_faults_to_data(self, data, label):
        label = label.numpy()
        for lab in label:
            if not isinstance(lab, int):
                lab = int(lab)
            add = [0, 0, 0, 0, 1, 0]
            data[lab][:-1] += add
        return data
    
    def add_no_failure_total(self, data):
        add = [0, 0, 0, 0, 1, 0] 
        data[5][:-1] += add
        
        return data
    
    def sum_labeled_results(self, data):
        sum_result = copy.deepcopy(self.label_data_struct)
        for label in data:
            sum_result[:-1] += data[label][:-1]
        
        return tuple(sum_result)
    
    def add_data_to_data(self, data, data_2):
        for label in data:
            data[label][:-1] += data_2[label][:-1]
            data[label][-1] = np.append(data[label][-1], data_2[label][-1])
        
        return data
    
    def add_correct_labelled_label(self, data, label):
        if not isinstance(label, int):
                label = int(label)
        
        add = [0, 0, 0, 0, 0, 1]
        data[label][:-1] += add
        return data
    
    def get_confusion_matrix_values(self, data, label="all"):
        if label == "all":
            return self.sum_labeled_results(data)
        elif label.lower() == "crack a":
            return tuple(data[1])
        elif label.lower() == "crack b":
            return tuple(data[2])
        elif label.lower() == "crack c":
            return tuple(data[3])
        elif label.lower() == "finger failure":
            return tuple(data[4])
        elif label.lower() == "no failure":
            return tuple(data[5])

    def initialize_data(self):
        data = {1: copy.deepcopy(self.label_data_struct), 
                2: copy.deepcopy(self.label_data_struct), 
                3: copy.deepcopy(self.label_data_struct), 
                4: copy.deepcopy(self.label_data_struct),
                5: copy.deepcopy(self.label_data_struct)}
        
        return data
    
    def calculate_accuracy_confusion_matrix(self, Tpos, Tneg, Fpos, Fneg):
        return (Tpos+Tneg)/(Tpos+Tneg+Fpos+Fneg)
         
    def print_data(self, data, success_percent, label="all"):
        label_mapper = LabelEncoder(binary=False)
        Tpos, Tneg, Fpos, Fneg, total_faults, correct_faults, Fpos_labels = self.get_confusion_matrix_values(data, label)
        accuracy = self.calculate_accuracy_confusion_matrix(Tpos, Tneg, Fpos, Fneg)
        print(f'-----------------------------------------------------------------------')
        print(f'Printing data for prediction of {label} label(s)')
        print(f'Tpos: {Tpos}')
        print(f'Tneg: {Tneg}')
        try:
            print(f'Targets found: {correct_faults/total_faults}')
        except:
            """Do nothing"""
        print(f'Targets total: {total_faults}')
        print(f'Mean accuracy: {accuracy}')
        print(f'Correctly found and labelled {label}: {correct_faults}')
        for key in data:
            if key == label_mapper.encode(label):
                continue
            print(f'{label_mapper.decode(key)}: {self.get_number_of_labels_predictions(key, Fpos_labels)}')
        
        print(f'False positive {label}: {Fpos}')
        print(f'-----------------------------------------------------------------------')

    def get_number_of_labels_predictions(self, label, fpos_label_array):
        return np.sum(fpos_label_array == label)
        
    def calc_accuracy(self, score, overlap_limit=0.3):
        
        # Label data {label: [Tpos, Tneg, Fpos, Fneg]}
        data = self.initialize_data()
        
        if score == 0:
            n_preds = 0
        else:
            try:
                n_preds = score.size  # Amount of predictions made
            except:
                n_preds = len(score)
                
        assert(len(self.tar) == 5), "Only one target at a time"  # Amount of target predictions
        
        n_targets = len(self.tar["boxes"])
        boxes_pred = self.img["boxes"].numpy()
        boxes_targ = self.tar["boxes"].numpy()
        
        data = self.add_total_faults_to_data(data, self.tar["labels"])

        # Stores success of target prediction
        predict_success = np.zeros(n_preds)
        target_success = np.zeros((n_targets)) 
        
        # First the positives
        for i in range(n_preds):
            current_pred = self.img["labels"][i]
            already_fpos_found = False
            for j in range(n_targets):
                current_targ = self.tar["labels"][j]
                overlap = self._calc_overlap(boxes_pred[i], boxes_targ[j])
            
                if overlap > overlap_limit:
                    if torch.all(current_pred.eq(current_targ)):
                        if target_success[j] != True:
                            data = self.add_correct_labelled_label(data, current_targ)
                            
                        data = self.add_stat_to_data(data, current_pred, 'Tpos')
                            
                        target_success[j] = True
                        predict_success[i] = True
                    else:
                        data = self.add_stat_to_data(data, current_pred, 'Fpos')
                        data = self.add_wrong_pred_label_to_data(data, current_pred, current_targ)
                        already_fpos_found = True
                    
                    break
        
            if not predict_success[i] and not already_fpos_found:
                data = self.add_stat_to_data(data, current_pred, 'Fpos')
                data = self.add_no_failure_total(data)
                data = self.add_wrong_pred_label_to_data(data, current_pred, 5)
                    
        n_fails = len(boxes_pred)-n_preds
        predict_failure = np.zeros(n_fails)

        # Then the negatives
        for k in range(n_fails):
            i = k + n_preds
            current_pred = 5
            for j in range(n_targets):
                current_targ = self.tar["labels"][j]
                
                overlap = self._calc_overlap(boxes_pred[i], boxes_targ[j])
                
                if overlap > overlap_limit:
                    data = self.add_wrong_pred_label_to_data(data, current_pred, current_targ)
                    data = self.add_stat_to_data(data, current_pred, 'Fneg')
                    break
                
                if j == (n_targets-1):
                    predict_failure[k] = True  
                    
            if predict_failure[k] == True:
                data = self.add_correct_labelled_label(data, current_pred)  
                data = self.add_no_failure_total(data)
                data = self.add_stat_to_data(data, current_pred, 'Tneg')

        summed_results = self.sum_labeled_results(data)
        
        assert(summed_results[4] >= len(self.tar["labels"]))
        
        
        return data, target_success, predict_success


    def get_success_w_box_overlap(self, score, overlap_limit=0.3):

        if score == 0:
            return None
        else:
            try:
                n_preds = score.size  # Amount of predictions made
            except:
                n_preds = len(score)
            n_targets = len(self.tar["labels"])  # Amount of target predictions

            boxes_pred = self.img["boxes"].numpy()
            boxes_targ = self.tar["boxes"].numpy()

            # Stores success of target prediction
            target_success = np.zeros(n_targets)
            predict_success = np.zeros(n_preds)

            # Determine if prediction is correct
            for i in range(n_preds):
                # Extract prediction labels
                current_pred = self.img["labels"][i]

                # Loop over targets
                for j in range(n_targets):
                    # Extract target labels
                    current_targ = self.tar["labels"][j]

                    # If prediction has already been linked to target, skip this iteration
                    if predict_success[i] != 0 or target_success[j] != 0:
                        continue

                    # If labels of prediction and target are equal, move on
                    if torch.all(current_pred.eq(current_targ)):
                        # Calculate overlap
                        overlap = self._calc_overlap(boxes_pred[i], boxes_targ[j])
                        # If bounding boxes also overlap enough, the prediction is a success
                        if overlap > overlap_limit:
                            target_success[j] = i
                            predict_success[i] = j

                            # print(f"Correct prediction of label {current_targ} with {overlap} percent overlap")

            return target_success, predict_success

    """---------------------------
    ---------BOX HELPERS----------
    ------------------------------"""

    def _get_box_center(self, boxes):
        xc = (boxes[2] / 2 + boxes[0] / 2).astype(np.uint64)
        yc = (boxes[3] / 2 + boxes[1] / 2).astype(np.uint64)
        return xc, yc

    def _calc_overlap(self, a, b):
        union = self._union(a, b)
        intersection = self._intersection(a, b)
        percent_area = self._calc_box_area(intersection) / self._calc_box_area(union)
        return percent_area

    def _calc_box_area(self, box):
        x, y, w, h = box
        xmin = x
        xmax = x + w
        ymin = y
        ymax = y + h
        area = (ymax - ymin) * (xmax - xmin)
        return area

    def _union(self, a, b):
        x = min(a[0], b[0])
        y = min(a[1], b[1])
        w = max(a[0] + a[2], b[0] + b[2]) - x
        h = max(a[1] + a[3], b[1] + b[3]) - y
        return (x, y, w, h)

    def _intersection(self, a, b):
        x = max(a[0], b[0])
        y = max(a[1], b[1])
        w = min(a[0] + a[2], b[0] + b[2]) - x
        h = min(a[1] + a[3], b[1] + b[3]) - y
        if w < 0 or h < 0:
            return (0, 0, 0, 0)
        return (x, y, w, h)

    """---------------------------
    ---------SECTION END----------
    ------------------------------"""

class LabelEncoder:

    def __init__(self, binary=False):

        if binary:
            self.fault_value_to_key = {'Crack A': 1,
                                       'Crack B': 1,
                                       'Crack C': 1,
                                       'Finger Failure' : 1}
        
        else:
            self.fault_value_to_key = {'Crack A': 1,
                                       'Crack B': 2,
                                       'Crack C': 3,
                                       'Finger Failure' : 4,
                                       'No failure' : 5}

    

        self.fault_key_to_value = {v: k for k, v in self.fault_value_to_key.items()}


    def encode(self,input):
        return self.fault_value_to_key[input]

    def decode(self,input):
        return self.fault_key_to_value[input]

