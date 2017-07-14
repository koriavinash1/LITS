import os
import time
import argparse
import shutil
import tensorflow as tf
from tf_layers import DFCN
from compute_metrics.metrics_acdc import compute_metrics_on_directories

def configure(isTrain=True):
    # training
    # TODO: configurations relevant
    flags = tf.app.flags
    # Train
    flags.DEFINE_integer('num_epochs', 250, '# Number of epoch')
    flags.DEFINE_integer('patience', 50, '# Patience parameter')
    flags.DEFINE_integer('save_step', 25, '# Save model at intervals')
    # Network-optimization
    flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
    flags.DEFINE_float('weight_decay', 0.0001, 'L2-Weight Decay parameter')
    # Data dependant
    flags.DEFINE_integer('n_classes', 4, 'output class number')
    if isTrain:
        flags.DEFINE_integer('batch_size', 10, 'batch size')
    else:
        flags.DEFINE_integer('batch_size', None, 'batch size')
    flags.DEFINE_integer('patient_batch_size', 1, 'Number of patients to prepare minibatches')
    flags.DEFINE_integer('channels', 1, 'channel size')
    flags.DEFINE_integer('height', 128, 'height size')
    flags.DEFINE_integer('width', 128, 'width size')
    # Model saving directories 
    flags.DEFINE_string('logdir', './DFCN/logdir', 'Log dir')
    flags.DEFINE_string('modeldir', './DFCN/modeldir', 'Model dir')
    flags.DEFINE_string('sample_dir', './DFCN/samples/', 'Sample directory')
    flags.DEFINE_string('best_model_dir', './DFCN/best_model/', 'Best model directory')
    flags.DEFINE_string('model_name', 'model', 'Model file name')
    flags.DEFINE_integer('test_epoch', 0, 'Test or predict epoch')
    #Debug
    flags.DEFINE_integer('random_seed', 0, 'random seed') 
    flags.DEFINE_integer('reload_epoch', 0, 'Reload epoch')
    # Network: DFCN:
    flags.DEFINE_integer('growth_rate', 8, 'Growth rate of filters in dense block')
    flags.DEFINE_integer('n_pool', 3, 'Number of dense blocks')
    flags.DEFINE_float('dropout_p', 0.2, 'dropout rate')   
    #Data: 
    flags.DEFINE_string('data_dir', 
        '/home/bmi/Documents/mak/Cardiac_dataset/ACDC/dataset/processed_data/pickled', 'Name of data directory')
    flags.DEFINE_string('full_data', 'complete_patient_data.pkl', 'Training data')
    flags.DEFINE_string('train_data', 'train_patient_data.pkl', 'Training data')
    flags.DEFINE_string('valid_data', 'validation_patient_data.pkl', 'Validation data')
    flags.DEFINE_string('test_data', 'test_patient_data.pkl', 'Testing data')
    flags.FLAGS.__dict__['__parsed'] = False
    return flags.FLAGS

def train():
    model = DFCN(tf.Session(), configure())
    model.train()


def test():
    pass

def predict():
    path='./predictions' 
    if os.path.exists(path):
        shutil.rmtree(path)
    gt_path = os.path.join(path,'gt')
    pred_path = os.path.join(path,'pd')
    os.makedirs(gt_path)
    os.makedirs(pred_path)

    model = DFCN(tf.Session(), configure(isTrain=False))
    model.predict(path)
    compute_metrics_on_directories(pred_path, gt_path)

def main(_):
    start = time.clock()
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', dest='action', type=str, default='train',
                        help='actions: train, test, or predict')
    args = parser.parse_args()
    if args.action not in ['train', 'test', 'predict']:
        print('invalid action: ', args.action)
        print("Please input a action: train, test, or segment")
    # test the model: evaluate segmentation metrics
    elif args.action == 'test':
        test()
    # Predict the segmentation maps
    elif args.action == 'predict':
        predict()
    # train
    else:
        train()
    end = time.clock()
    print("program total running time",(end-start)/60)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    tf.app.run()
    