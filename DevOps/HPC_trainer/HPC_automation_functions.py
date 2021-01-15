from DevOps.HPC_trainer.conf import ssh_conf as conf_file
import paramiko
from scp import SCPClient
import os
import sys
import time
import socket
import getpass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# print("current directory is:" + os.getcwd())


class SSH_Util:
    """Class to connect to remote HPC"""

    def __init__(self):
        self.ssh_output = None
        self.ssh_error = None
        self.client = None
        self.session = None
        self.transfer_client = None
        self.transfer_host = conf_file.TRANSFER_HOST
        self.host = conf_file.HOST
        self.host2 = conf_file.HOST2
        self.timeout = float(conf_file.TIMEOUT)
        self.commands = conf_file.COMMANDS
        self.port = conf_file.PORT

        # Setup related HPC commands
        self.pwd = None
        self.default_dir = conf_file.DEFAULT_ENV
        self.init_command = conf_file.INIT_COMMMAND

        self.username = input("Please input DTU username: ")
        self.password = getpass.getpass()
        self.scp_client = None

    def get_environment_path(self):
        return self.pwd.strip() + ("/" + self.default_dir).encode()

    def connect_transfer_client(self):
        print("Connected to server", self.host)
        self.scp_client = SCPClient(self.client.get_transport(), progress=self.progress)

    def connect_execution_client(self, host):
        self.client.connect(
            hostname=host,
            port=self.port,
            username=self.username,
            password=self.password,
            timeout=self.timeout,
            allow_agent=False,
            look_for_keys=False,
        )

    def connect_client(self):
        try:
            self.connect_execution_client(self.host)

            print("Connected to server", self.host)
            self.connect_transfer_client()
            return True
        except:
            print("First host was not able to connect")

        try:
            self.connect_execution_client(self.host2)
            print("Connected to server", self.host2)
            self.connect_transfer_client()
            return True
        except:
            print("Second host was not able to connect either")
            return False

    def connect(self):
        """Login"""
        try:
            print("Establishing connection...")
            self.transfer_client = paramiko.SSHClient()
            self.transfer_client.set_missing_host_key_policy(
                paramiko.AutoAddPolicy())
            self.client = paramiko.SSHClient()
            self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            self.connect_client()

        except paramiko.AuthenticationException:
            print("Authentication failed, please verify your credentials")
            result_flag = False
        except paramiko.SSHException as sshException:
            print("Could not establish SSH connection: %s" % sshException)
            result_flag = False
        except socket.timeout as e:
            print("Connection timed out")
            result_flag = False
        except Exception as e:
            print("\nException in connecting to the server")
            print("PYTHON SAYS:", e)
            result_flag = False
            self.client.close()
        else:
            result_flag = True

        return result_flag

    def execute_command(self, commands, init=False, print_output=False):
        """Execute a command on the remote host. Return a tuple containing
        an integer status and a two strings, the first containing stdout
        and the second containing stderr from the command."""
        self.ssh_output = None
        result_flag = True

        try:
            commands = self.create_command(commands, init)
            print("Executing command --> {}".format(commands))
            stdin, stdout, stderr = self.client.exec_command(
                commands, get_pty=True, timeout=10
            )
            self.ssh_output = stdout.read()
            self.ssh_error = stderr.read()

            i = 0
            if self.ssh_error:
                if i > 0:
                    print(
                        "Problem occurred while running command:"
                        + commands
                        + " The error is "
                        + self.ssh_error.decode("utf-8")
                    )
                    result_flag = False
            else:
                print("Command execution completed successfully", commands)
                if print_output:
                    print(str(self.ssh_output, "utf-8"))

        except socket.timeout as e:
            print("Command timed out.", commands)
            self.client.close()
            result_flag = False
        except paramiko.SSHException:
            print("Failed to execute the command!", commands)
            self.client.close()
            result_flag = False

        return result_flag, self.ssh_output

    def download_file(self, downloadremotefilepath, downloadlocalfilepath):
        """This method downloads the file from remote server"""
        result_flag = True
        try:
            if self.connect():
                ftp_client = self.client.open_sftp()
                ftp_client.get(downloadremotefilepath, downloadlocalfilepath)
                ftp_client.close()
                self.client.close()
            else:
                print("Could not establish SSH connection")
                result_flag = False
        except Exception as e:
            print(
                "\nUnable to download the file from the remote server",
                downloadremotefilepath,
            )
            print("PYTHON SAYS:", e)
            result_flag = False
            ftp_client.close()
            self.client.close()

        return result_flag

    def ssh_init(self):
        self.connect()
        self.execute_command(["pwd"])
        self.pwd = self.ssh_output

    def does_data_directory_exists(self):
        decode_output = {"1": True, "": False}
        env_path = str(self.get_environment_path().strip(), "utf-8")
        _, command_output = self.execute_command(
            [f'[ -d "{env_path}/data" ] && echo "1" '], init=False
        )
        return decode_output[str(command_output.strip(), "utf-8")]

    def scp_necessary_files(self, force_upload_data=False):
        if not self.does_data_directory_exists() or force_upload_data:
            self.scp_client.put(
                "data", recursive=True, remote_path=self.get_environment_path()
            )

        # Overwriting existing files
        self.scp_client.put(
            "components", recursive=True, remote_path=self.get_environment_path()
        )
        self.scp_client.put(
            "img-inspect.py", recursive=True, remote_path=self.get_environment_path()
        )
        self.scp_client.put(
            "train.py", remote_path=self.get_environment_path())
        self.scp_client.put("requirements-cuda-10.1.txt",
                            remote_path=self.get_environment_path())
        self.scp_client.put(
            "DevOps/HPC_trainer/conf/QueJob.sh", remote_path=self.get_environment_path()
        )
        self.scp_client.put("model_conf.json",
                            remote_path=self.get_environment_path())

    def create_command(self, commands, init=True):
        if init:
            output_command = self.init_command
        else:
            output_command = ""

        for command in commands:
            if output_command == "":
                output_command = output_command + command
            else:
                output_command = output_command + ";" + command

        return output_command

    def que_job(self):
        self.scp_necessary_files()
        print("Queing job")
        self.execute_command(
            [f"cd {self.default_dir};bsub < QueJob.sh"],
            init=True,
            print_output=True,
        )
    
    def progress(self, filename, size, sent):
        sys.stdout.write("%s's progress: %.2f%%   \r" % (filename, float(sent)/float(size)*100) )
    
    def extract_results_folder(self):
        self.scp_client.get(remote_path=self.get_environment_path() + b"/Results-folder", recursive=True)
    
    def close_session(self):
        self.client.close()


# Working example that connects to HPC and executes simple commands
if __name__ == "__main__":
    print("Start of %s" % __file__)

    # Initialize the ssh object
    ssh_obj = SSH_Util()
    ssh_obj.ssh_init()
    ssh_obj.que_job()
    ssh_obj.close_session()
