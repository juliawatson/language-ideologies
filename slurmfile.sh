#!/bin/bash
#SBATCH --job-name=lang_ideo_experiment
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --qos=normal
#SBATCH --time=14:00:00
#SBATCH -c 4
#SBATCH --mem=20G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# prepare your environment here
source /h/jwatson/some_env/bin/activate

# put your command here
# python part_2_query_llama.py
python part_2_query_t5.py
