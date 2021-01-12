

class LabelEncoder:

    def __init__(self):

    def fault_value_to_key(self, binary = False):
        if binary:
            fault = {'Crack A': 1,
                     'Crack B': 1,
                     'Crack C': 1,
                     'Finger Failure' : 1,
                     'No failure' : 2}
        else:
            fault = {'Crack A': 1,
                     'Crack B': 2,
                     'Crack C': 3,
                     'Finger Failure' : 4,
                     #'No failure' : 2
                     }


    def fault_key_to_value(self):
        fault = {v: k for k, v in fault_value_to_key.items()}


    def encode(self):
        return fault_value_to_key()

    def decode(self):
        return fault_key_to_value()

def main():


if __name__ == "__main__":
    main()