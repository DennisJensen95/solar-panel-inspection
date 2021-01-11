from DevOps.HPC_trainer.HPC_automation_functions import SSH_Util
from components.model_configuration.configuration import ConfActionStepOne
import argparse as ap
import time
import os


def main():
    parser = ap.ArgumentParser(description="Training script")
    parser.add_argument("--Multiple")
    args, leftovers = parser.parse_known_args()
    configuration_class = ConfActionStepOne()

    # Initialize SSH connection
    ssh_connection = SSH_Util()
    ssh_connection.ssh_init()

    if args.Multiple != None:
        list_of_configurations = configuration_class.get_configurations()

        for conf in list_of_configurations:
            path = "model_conf.json"
            if os.path.exists(path):
                os.remove(path)
            time.sleep(3)
            configuration_class.write_configuration_file(path, conf)
            time.sleep(3)
            print(f"Configuration setup: {conf}")
            ssh_connection.que_job()
            time.sleep(2)
    else:
        ssh_connection.que_job()

    ssh_connection.close_session()


if __name__ == "__main__":
    main()
