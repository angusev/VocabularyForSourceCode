#!/bin/sh

#SBATCH --error=./slurm/%j.err --output=./slurm/%j.out -J c2s
#SBATCH -G 1 -c 4

type=java-small-model
dataset_name=java-small
data_dir=data/java-small
data=${data_dir}/${dataset_name}
test_data=${data_dir}/${dataset_name}.val.c2s

random4="$(openssl rand -hex 4)"
currentDate="`date +"%Y_%m_%d"`"
lasso="$1"
group_lasso="$2"
threshold="$3"
model_dir=models/${type}/${currentDate}__$1_$2_$3__${random4}

mkdir -p ${model_dir}
set -e
python3 -u code2seq.py --data ${data} --test ${test_data} --save_prefix ${model_dir}/model --lasso $1 --grouplasso $2 --threshold $3
