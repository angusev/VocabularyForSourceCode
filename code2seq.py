from argparse import ArgumentParser
import numpy as np
import tensorflow as tf

from config import Config
from interactive_predict import InteractivePredictor
from model import Model


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
    parser.add_argument("-d", "--data", dest="data_path",
                        help="path to preprocessed dataset", required=False)
    parser.add_argument("-te", "--test", dest="test_path",
                        help="path to test file", metavar="FILE", required=False)

    parser.add_argument("-s", "--save_prefix", dest="save_path_prefix",
                        help="path to save file", metavar="FILE", required=False)
    parser.add_argument("-l", "--load", dest="load_path",
                        help="path to saved file", metavar="FILE", required=False)
    parser.add_argument('--release', action='store_true',
                        help='if specified and loading a trained model, release the loaded model for a smaller model '
                             'size.')
  
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=239)

    parser.add_argument('--lasso', action='store', default=0, type=float, help='L1-regularisation on embeddings layer coefficient')
    parser.add_argument('--grouplasso', action='store', default=0, type=float, help='Group Lasso regularisation on embeddings layer coefficient')
    parser.add_argument('--threshold', action='store', default=-1, type=float, help='Threshold applying for reseting values of tensors to zeros')
    
    parser.add_argument('--subtoken_words', action='store', default=190000, type=int, help='SUBTOKEN_VOCAB words max number restriction')
    parser.add_argument('--nodes_words', action='store', default=-1, type=int, help='NODES_VOCAB words max number restriction')
    parser.add_argument('--sparse_nodes', type=str2bool, default=True,  help="Flag responcing for NODES_VOCAB embeddings sparsification")
    parser.add_argument('--sparse_subtoken', type=str2bool, default=True,  help="Flag responcing for SUBTOKEN_VOCAB embeddings sparsification")

    args = parser.parse_args()
    
    if args.nodes_words == -1:
        args.nodes_words = None

    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    if args.debug:
        config = Config.get_debug_config(args)
    else:
        config = Config.get_default_config(args)

    model = Model(config)
    print('Created model')
    if config.TRAIN_PATH:
        model.train()
    if config.TEST_PATH and not args.data_path:
        results, precision, recall, f1, rouge = model.evaluate()
        print('Accuracy: ' + str(results))
        print('Precision: ' + str(precision) + ', recall: ' + str(recall) + ', F1: ' + str(f1))
        print('Rouge: ', rouge)
    if args.predict:
        predictor = InteractivePredictor(config, model)
        predictor.predict()
    if args.release and args.load_path:
        model.evaluate(release=True)
    model.close_session()
