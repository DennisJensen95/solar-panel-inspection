HOST = "login1.hpc.dtu.dk"
HOST2 = "login2.hpc.dtu.dk"
TRANSFER_HOST = "transfer.gbar.dtu.dk"
PORT = 22
TIMEOUT = 20
DEFAULT_ENV = "solar_inspect"

# Sample commands to execute
INIT_COMMMAND = "source /etc/profile;source ~/.bashrc;module load python3/3.8.2;module load cuda/10.1;module load cudnn/v8.0.4.30-prod-cuda-10.1"
MAKE_ENV = f"mkdir -p {DEFAULT_ENV};cd {DEFAULT_ENV}"
GET_PWD = "pwd"

COMMANDS = [INIT_COMMMAND, MAKE_ENV, GET_PWD]

# # Sample file location to upload and download
# UPLOAD_REMOTEFILE_PATH = '/dronel_project/img-inspect.py'
# UPLOAD_LOCALFILE_PATH = '/../filename.txt'
# DOWNLOAD_REMOTEFILE_PATH = '/etc/sample/data.txt'
# DOWNLOAD_LOCALFILE_PATH = 'home/data.txt'
