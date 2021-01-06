# Solar panel inspection software stack

Solar inspection software

## Environment analytics

The framework for training the deep learning model is pytorch.

We apply transfer learning to achieve a good model for feature recognition on the images.

### HPC automation

Python script for HPC automation w.r.t training of neural network

To automatically que a training job in HPC run the following command:

```console
solar-panel-inspection$ cd inspection_software
solar-panel-inspection$ python3 DevOps/HPC_trainer/HPC_automation_functions.py
Start of DevOps/HPC_trainer/HPC_automation_functions.py
Please input DTU username: s155629
Password:
Establishing connection...
First host was not able to connect
Connected to server login2.hpc.dtu.dk
Connected to server login1.hpc.dtu.dk
Executing command --> pwd
Command execution completed successfully pwd
Executing command --> [ -d "/zhome/9b/c/111496/solar_inspect/data" ] && echo "1"
Command execution completed successfully [ -d "/zhome/9b/c/111496/solar_inspect/data" ] && echo "1"
Queing job
Executing command --> source /etc/profile;source ~/.bashrc;module load python3/3.8.2;module load cuda/10.1;module load cudnn/v8.0.4.30-prod-cuda-10.1;cd solar_inspect;ls;bsub < QueJob.sh
Command execution completed successfully source /etc/profile;source ~/.bashrc;module load python3/3.8.2;module load cuda/10.1;module load cudnn/v8.0.4.30-prod-cuda-10.1;cd solar_inspect;ls;bsub < QueJob.sh
Loaded dependency [gcc/8.4.0]: binutils/2.34
Loaded dependency [python3/3.8.2]: gcc/8.4.0
Loaded module: python3/3.8.2
Loaded module: cuda/10.1
Loaded module: cudnn/v8.0.4.30-prod-cuda-10.1
components  data  QueJob.sh  Results-Folder  train.py
Job <8635177> is submitted to queue <gpuv100>.

```

Thus the job will que and execute.