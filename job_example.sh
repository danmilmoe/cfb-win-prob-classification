#!/bin/bash
#SBATCH --job-name=ijustworkhere
#SBATCH --account=eecs448w24_class
#SBATCH --partition=standard
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16gb
#SBATCH --mail-type=END

module purge

pip3 install --user pandas

python3 sort_by_thread.py
