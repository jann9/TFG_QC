#!/usr/bin/env bash
# Leave only one comment symbol on selected options
# Those with two commets will be ignored:
# The name to show in queue lists for this job:
#SBATCH -J ML_QAOA

# Number of desired cpus (can be in any node):
##SBATCH --ntasks=1

# Number of desired cpus (all in same node):
#SBATCH --cpus-per-task=5

# Amount of RAM needed for this job:
#SBATCH --mem=8gb

# The time the job will be running:
#SBATCH --time=168:00:00

# To use GPUs you have to request them:
##SBATCH --gres=gpu:1

# If you need nodes with special features uncomment the desired constraint line:
# * to request only the machines with 80 cores and 2TB of RAM
##SBATCH --constraint=bigmem
# * to request only machines with 16 cores and 64GB with InfiniBand network
#SBATCH --constraint=cal
# * [sr]: 156 x Lenovo SR645 nodes: 128 cores (AMD EPYC 7H12 @ 2.6GHz), 512 GB of RAM. InfiniBand HDR100 network.. 900 GB of localscratch disks.
##SBATCH --constraint=dx
##SBATCH --constraint=ssd
##SBATCH --constraint=sr
#SBATCH --constraint=cal

# Set output and error files
#SBATCH --error=logs/job.%J.err
#SBATCH --output=logs/job.%J.out

# Leave one comment in following line to make an array job. Then N jobs will be launched. In each one SLURM_ARRAY_TASK_ID will take one value from 1 to 100
#SBATCH --array=1-32

# To load some software (you can show the list with 'module avail'):
# module load software


# the program to execute with its parameters:
mkdir logs
mkdir -p Models/ml_vs_ml/mlp
mkdir -p Models/ml_vs_ml/gnn
mkdir -p Models/ml_vs_ml/xgboost
source ~/environement/env_ml_qaoa/bin/activate
origin=`pwd`
dir_temp=${LOCALSCRATCH}${USER}/${SLURM_JOB_ID}/job_${SLURM_ARRAY_TASK_ID}
mkdir -p ${dir_temp}/Models/ml_vs_ml/mlp
mkdir -p ${dir_temp}/Models/ml_vs_ml/gnn
mkdir -p ${dir_temp}/Models/ml_vs_ml/xgboost
mkdir -p ${dir_temp}/datasets
cp -r datasets/* ${dir_temp}/datasets/
cp -r *.py ${dir_temp}
cd ${dir_temp}
ml_tech=('mlp' 'xgboost' 'gnn')
for tech in "${ml_tech[@]}"
do
  python model_train_${tech}.py
done
zip -r execution_${SLURM_ARRAY_TASK_ID}.zip Models/ml_vs_ml/*
mv execution_${SLURM_ARRAY_TASK_ID}.zip ${origin}/Models/ml_vs_ml
