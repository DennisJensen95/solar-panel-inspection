from DevOps.HPC_trainer.HPC_automation_functions import SSH_Util
from components.model_configuration.configuration import ConfActionStepOne
import argparse as ap
import time
import os


def main():
    parser = ap.ArgumentParser(description="Training script")
    args, leftovers = parser.parse_known_args()
    configuration_class = ConfActionStepOne()

    # Initialize SSH connection
    ssh_connection = SSH_Util()
    ssh_connection.ssh_init()
    ssh_connection.extract_results_folder()
    ssh_connection.close_session()


if __name__ == "__main__":
    main()
