import os
import torch.nn as nn
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter


def to_np(x):
    return x.data.cpu().numpy()


class TensorBoardLogger(object):

    def __init__(self, id, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        self.id = id
        self.tensorboard_writer = SummaryWriter(log_dir)

    def update(self, epoch, values):
        values = {
            'Avg. Train Loss': values["loss_results"][epoch],
            'WER': values["wer"][epoch],
            'CER': values["cer"][epoch]
        }

        self.tensorboard_writer.add_scalars(self.id, values, epoch + 1)

    def load_previous_values(self, start_epoch, values):
        for i in range(start_epoch):
            values = {
                'Avg. Train Loss': values["loss_results"][i],
                'WER': values["wer"][i],
                'CER': values["cer"][i]
            }

            self.tensorboard_writer.add_scalars(self.id, values, i + 1)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= world_size
    return rt


def sum_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    return rt


def get_default_audio_config():
    return {
        "normalize": True,
        "sampling_rate": 16000,
        "window": "hamming",
        "window_stride": 0.01,
        "window_size": 0.02
    }


def serialize(model, optimizer=None, epoch=None, iteration=None, loss_results=None,
              cer_results=None, wer_results=None, avg_loss=None, meta=None, distributed=False,
              streaming_model=None, context=None):

    supported_rnns = {
        'lstm': nn.LSTM,
        'rnn': nn.RNN,
        'gru': nn.GRU
    }
    supported_rnns_inv = dict((v, k) for k, v in supported_rnns.items())

    if distributed:
        package = {
            'model_name': model.module.model_name,
            'conv_layers': model.module.conv_layers,
            'rnn_hidden_size': model.module.rnn_hidden_size,
            'rnn_layers': model.module.rnn_layers,
            'rnn_type': supported_rnns_inv.get(model.module.rnn_type, model.module.rnn_type.__name__.lower()),
            'audio_conf': model.module.audio_conf,
            'labels': model.module.labels,
            'bidirectional': model.module.bidirectional,
        }
    else:
        package = {
            'model_name': model.model_name,
            'conv_layers': model.conv_layers,
            'rnn_hidden_size': model.rnn_hidden_size,
            'rnn_layers': model.rnn_layers,
            'rnn_type': supported_rnns_inv.get(model.rnn_type, model.rnn_type.__name__.lower()),
            'audio_conf': model.audio_conf,
            'labels': model.labels,
            'state_dict': model.state_dict(),
            'bidirectional': model.bidirectional
        }

    if optimizer is not None:
        package['optim_dict'] = optimizer.state_dict()
    if avg_loss is not None:
        package['avg_loss'] = avg_loss
    if epoch is not None:
        package['epoch'] = epoch + 1
    if iteration is not None:
        package['iteration'] = iteration
    if loss_results is not None:
        package['loss_results'] = loss_results
        package['cer_results'] = cer_results
        package['wer_results'] = wer_results
    if meta is not None:
        package['meta'] = meta
    if streaming_model is not None:
        package['streaming_model'] = streaming_model
    if context is not None:
        package['context'] = context
    return package

# ----- Johan: Logger -----
import os
import json
import pandas as pd

def loadLogger():
    # Load saved scores
    if 'scoreLogger.json' in os.listdir('Scores/'):
        with open('Scores/scoreLogger.json') as f:
            scores = json.load(f)
            
    # Else: Create new empty score dictionary
    else:
        scores = {}
        
    return scores

def saveScores(scoresDict):
    # Save scores to local directory
    with open('Scores/scoreLogger.json', 'w') as f:
        json.dump(scoresDict, f, indent=4)
    print('Scores saved...')
    return

def loadCSV():
    # Load saved scores
    path = '/home/user/DanSpeech-AdvancedMachineLearning/AML Cleaned/Scores'
    if 'allScores.csv' in os.listdir(path):
        df = pd.read_csv('Scores/allScores.csv', sep=";")
            
    # Else: Create new empty score dictionary
    else:
        # - Tue. Time added
        columns = ['Time','score_ID', 'Reference', 'WER', 'CER', 'Transcript', 'Augmentation_list',"Best_epoch","Total_epoch"]
        df = pd.DataFrame(columns=columns)
        
    return df

def csvSaver(lst):
    '''
    Takes a nested list as input.
    Saves a csv file.
    '''
    # Open saved data
    MainDF = loadCSV()
    

    # Create new df
    # - Tue: Time added
    columns = ['Time','score_ID', 'Reference', 'WER', 'CER', 'Transcript', 'Augmentation_list',"Best_epoch","Total_epoch"]
    df = pd.DataFrame(lst, columns=columns)
    
    path = '/home/user/DanSpeech-AdvancedMachineLearning/AML Cleaned/Scores'
    
    # Merge the two dataframes
    MainDF = MainDF.append(df)
    MainDF.to_csv(os.path.join(path,'allScores.csv'), index=False, sep=";")
    print('Saving all scores into csv...')
    return

#Saving test scores to csv file
def loadCSVTest():
    # Load saved scores
    path = '/home/user/DanSpeech-AdvancedMachineLearning/AML Cleaned/Scores'
    if 'allScoresTest.csv' in os.listdir(path):
        df = pd.read_csv('Scores/allScoresTest.csv', sep=";")
            
    # Else: Create new empty score dictionary
    else:
        # - Tue. Time added
        columns = ["Data_type","Decoder_type",'LM_type','model_ID', 'Reference', 'WER', 'CER', 'Transcript']
        df = pd.DataFrame(columns=columns)
        
    return df

def csvSaverTest(lst):
    '''
    Takes a nested list as input.
    Saves a csv file.
    '''
    # Open saved data
    MainDF = loadCSVTest()
    
    # Create new df
    # - Tue: Time added
    columns = ["Data_type","Decoder_type",'LM_type','model_ID', 'Reference', 'WER', 'CER', 'Transcript']
    df = pd.DataFrame(lst, columns=columns)
    
    path = '/home/user/DanSpeech-AdvancedMachineLearning/AML Cleaned/Scores'
    
    # Merge the two dataframes
    MainDF = MainDF.append(df)
    MainDF.to_csv(os.path.join(path,'allScoresTest.csv'), index=False, sep=";")
    print('Saving all test scores into csv...')
    return