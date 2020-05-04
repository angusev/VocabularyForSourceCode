import os
import secrets
from datetime import datetime
from argparse import ArgumentParser


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
       
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--lasso', action='store', default=0, type=float, help='L1-regularisation on embeddings layer coefficient')
    parser.add_argument('--grouplasso', action='store', default=0, type=float, help='Group Lasso regularisation on embeddings \
            layer coefficient')
    parser.add_argument('--threshold', action='store', default=-1, type=float, help='Threshold applying for reseting values of tensors to zeros')
    
    parser.add_argument('--subtoken_words', action='store', default=190000, type=int, help='SUBTOKEN_VOCAB words max number restriction')
    parser.add_argument('--nodes_words', action='store', default=-1, type=int, help='NODES_VOCAB words max number restriction')
    parser.add_argument("--sparse_nodes", type=str2bool, nargs='?', const=False, default=True,  help="Flag responcing for NODES_VOCAB embeddings sparsification")
    parser.add_argument("--sparse_subtoken", type=str2bool, nargs='?', const=False, default=True,  help="Flag responcing for SUBTOKEN_VOCAB embeddings sparsification")
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
    subtoken_vocab_max_size = args.subtoken_words
    nodes_vocab_max_size = args.nodes_words
    
    model_dir = f'./models/{type}/{currentDate}__{round(lasso, 5)}_{round(grouplasso, 5)}_{round(threshold, 5)}__{random4}'

    os.system(f'mkdir -p {model_dir}')
    os.system('set -e')
    
    python_command = f'python3 -u code2seq.py --data {data}\
                                              --test {test_data}\
                                              --save_prefix {model_dir}/model\
                                              --lasso {lasso}\
                                              --grouplasso {grouplasso}\
                                              --threshold {threshold}\
                                              --subtoken_words {subtoken_vocab_max_size}\
                                              --nodes_words {nodes_vocab_max_size}\
                                              --sparse_nodes {args.sparse_nodes}\
                                              --sparse_subtoken {args.sparse_subtoken}'
    
    slurm_command = f'sbatch --error={model_dir}/%j.err --time=3-0:0 --output={model_dir}/%j.out -J c2s --gres=gpu:1 -c 4 --wrap=\"{python_command}\"'
    os.system(slurm_command)

