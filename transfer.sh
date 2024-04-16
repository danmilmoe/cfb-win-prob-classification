#!/bin/bash
#SBATCH --job-name=ijustworkhere
#SBATCH --account=eecs448w24_class
#SBATCH --partition=standard
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16gb

module purge

pip3 install --user pandas
pip3 install --user tensorflow
pip3 install --user scikit-learn
pip3 install --user keras
pip3 install --user numpy

python3 classify.py
