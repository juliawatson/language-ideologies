#!/bin/bash

# Julia: We should remove this before we release the code.
# I used this to refactor the repo so that the results were structured as:
#     analyses/experimenti/run_descriptor/model_name
# which supports having multiple models per run_descriptor

for dirname in /Users/julia/Documents/Fall2023/language_reform_2023/fall_2023_main/analyses/experiment2/*; do
    new_dir="${dirname}/text-davinci-003/"
    mkdir $new_dir;
    for filename in "${dirname}"/*; do
        base_name=$(basename ${filename})
        if [ $base_name != "stimuli.csv" ] && [ $base_name != "config.json" ] && [ $base_name != "stimuli_example.csv" ]
        then
            git mv ${filename} ${new_dir}
            # mv ${filename} ${new_dir}
        fi
    done
done