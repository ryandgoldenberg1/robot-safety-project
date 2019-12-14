#!/usr/bin/env bash

# The script has the commands used to setup GCP instance with software needed for this repo. 
sudo apt update
sudo apt install git tree python3-pip vim unzip libgl1-mesa-dev
sudo apt-get install libosmesa6-dev libglfw3 libglfw3-dev
sudo add-apt-repository ppa:jamesh/snap-support
sudo apt install patchelf
sudo apt-get remove python-cryptography
sudo apt-get remove python3-cryptography

git clone https://github.com/ryandgoldenberg1/robot-safety-project.git
cd robot-safety-project
pip3 install --no-cache-dir -r requirements.txt

# Install MuJoCo
curl https://www.roboti.us/getid/getid_linux --output getid_linux
chmod 744 getid_linux
./getid_linux
mkdir .mujoco
cd .mujoco
gsutil cp gs://safety-project-data/instance-2/mjkey.txt .
wget https://www.roboti.us/download/mujoco200_linux.zip
unzip mujoco200_linux.zip
mv mujoco200_linux mujoco200
rm mujoco200_linux.zip
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ryan/.mujoco/mujoco200/bin
pip3 install mujoco-py

# Install safety_gym
git clone https://github.com/openai/safety-gym.git
cd safety-gym
pip3 install -e .
