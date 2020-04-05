import os
import secrets
from datetime import datetime

from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--lasso', action='store', default=0, type=float, help='L1-regularisation on embeddings layer coefficient')
    parser.add_argument('--grouplasso', action='store', default=0, type=float, help='Group Lasso regularisation on embeddings \
            layer coefficient')
    parser.add_argument('--threshold', action='store', default=1e10, type=float, help='Threshold applying for reseting values of tensors \
            to zeros')


    args = parser.parse_args()

    type = 'java-small-model'
    dataset_name = 'java-small'
    data_dir = 'data/java-small'
    data = f'{data_dir}/{dataset_name}'
    test_data= f'{data_dir}/{dataset_name}.val.c2s'

    random4 = secrets.token_hex(4)
    currentDate =  datetime.now().strftime('%Y_%m_%d')
    lasso = args.lasso
    grouplasso = args.grouplasso
    threshold = args.threshold
    model_dir = f'./models/{type}/{currentDate}__{lasso}_{grouplasso}_{threshold}__{random4}'

    os.system(f'mkdir -p {model_dir}')
    os.system('set -e')
    
    python_command = f'python3 -u code2seq.py --data {data} --test {test_data} --save_prefix {model_dir}/model --lasso {lasso} --grouplasso {grouplasso} --threshold {threshold}'
    slurm_command = f'sbatch --error={model_dir}/%j.err --output={model_dir}/%j.out -J c2s --gres=gpu:1 -c 4 --wrap=\"{python_command}\"'
    os.system(slurm_command)


