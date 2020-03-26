type=java-small-model
dataset_name=java-small
data_dir=data/java-small
data=${data_dir}/${dataset_name}
test_data=${data_dir}/${dataset_name}.val.c2s
model_dir=models/${type}

mkdir -p ${model_dir}
set -e
python3 -u code2seq.py --data ${data} --test ${test_data} --save_prefix ${model_dir}/model --lasso $1 --grouplasso $2 --threshold $3
