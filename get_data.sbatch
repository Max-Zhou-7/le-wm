#!/bin/bash
#SBATCH -p short
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH -o get_data_%j.log
source ~/.bashrc
conda activate lewm
bash get_data.sh
