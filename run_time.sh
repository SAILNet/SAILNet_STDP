#!/bin/bash
# Job name
#SBATCH --job-name=SAILNet
#
# Partition:
#SBATCH --partition=cortex
#
# Wall clock limit:
#SBATCH --time=24:00:00
#
# Memory:
#SBATCH --mem-per-cpu=15G
#
# Constraint:
#SBATCH --constraint=cortex_fermi
#
module load cuda
module unload intel
cd $SLURM_SUBMIT_DIR

#Time Data and Keep Spikes
python SAILNet.py dW_SAILnet None -n 30000 --plot_interval 10000 --OC1 6 -s 5000 -d 30000 -k True -e True
python SAILNet.py dW_time_dep Gaussian -n 30000 --plot_interval 10000 --OC1 6 -s 5000 -d 30000 -k True -e True
#python SAILNet.py dW_time_dep Border_Gaussian -n 30000 --plot_interval 10000 --OC1 6 -s 5000 -b 30000 -k True -e True
#Time Data, Static Learning
python SAILNet.py dW_SAILnet None -n 30000 --plot_interval 10000 --OC1 6 -s 5000 -d 30000 -k True -e True -j True
python SAILNet.py dW_time_dep Gaussian -n 30000 --plot_interval 10000 --OC1 6 -s 5000 -d 30000 -k True -e True -j True
#python SAILNet.py dW_time_dep Border_Gaussian -n 30000 --plot_interval 10000 --OC1 6 -s 5000 -b 30000 -k True -e True -j True
#Time Data, Keep Spikes and Decay W
python SAILNet.py dW_SAILnet None -n 30000 --plot_interval 10000 --OC1 6 -s 5000 -d 30000 -k True -e True -w True
python SAILNet.py dW_time_dep Gaussian -n 30000 --plot_interval 10000 --OC1 6 -s 5000 -d 30000 -k True -e True -w True
#python SAILNet.py dW_time_dep Border_Gaussian -n 30000 --plot_interval 10000 --OC1 6 -s 5000 -b 30000 -k True -e True -w True
#Time Data, Static Learning and W Decay
python SAILNet.py dW_SAILnet None -n 30000 --plot_interval 10000 --OC1 6 -s 5000 -d 30000 -k True -e True -w True -j True
python SAILNet.py dW_time_dep Gaussian -n 30000 --plot_interval 10000 --OC1 6 -s 5000 -d 30000 -k True -e True -w True -j True
#python SAILNet.py dW_time_dep Border_Gaussian -n 30000 --plot_interval 10000 --OC1 6 -s 5000 -b 30000 -k True -e True -w True -j True
