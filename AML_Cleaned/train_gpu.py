import os
import sys
#os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
import torch
from deepspeech.train import finetune#, train_new, continue_training
from audio.dataset_factory import DatasetFactory
from sklearn.model_selection import KFold
if torch.cuda.is_available():
    print("Success")
    print(torch.cuda.device_count())
    

#Original room reverberation values    
augment_parameters = {'speed_perturb'  : [0.9,1.1],
                  'shift_perturb'  : [-50,50],
                  'room_reverb'    : [0, 0.4, 3.0, 0, 5, 3.0, 0, 5, 3.0, 0, 5, 0.5, 0.5, 0.5],
                  'volume_perturb' : [5,30],
                  'add_wn'         : [0.5,1.8],
                  'tempo_perturb'  : [-0.25,0.25],
                  'spec_augment'   : [80, 27, 50, 1, 1]
                 }

    
def f_train_gpu(model_name="NoName",augmentation_list=[],k_folds=None, epochs = 20, root_dir = '../Data/Tue_test_split/',augment_parameters = augment_parameters,augment_prob_dir=None):    


    factory = DatasetFactory(root_dir)
    
    augment_parameters = augment_parameters
    

    if k_folds:

        CV = KFold(k_folds, shuffle=True, random_state=42)

        for train_index, validation_index in CV.split(factory.meta):
            training_set = factory.meta.iloc[train_index]
            validation_set = factory.meta.iloc[validation_index]


            finetune(model_id=model_name,
                     root_dir=root_dir,
                     training_set=training_set,
                     validation_set=validation_set,
                     in_memory=True,
                     stored_model='/home/user/.danspeech/models/DanSpeechPrimary.pth',
                     augmentation_list=augmentation_list,
                     cuda=True,
                     augment_parameters = augment_parameters,
                     model_save_dir='/home/user/DanSpeech-AdvancedMachineLearning/AML Cleaned/model_save_dir/',
                     epochs = epochs,
                     augment_prob_dir = augment_prob_dir
            )


    else:

        factory.split_file(0.67)
        finetune(model_id=model_name,
                 root_dir=root_dir,
                 training_set='train.csv',
                 validation_set='validation.csv',
                 stored_model='/home/user/.danspeech/models/DanSpeechPrimary.pth',
                 augmentation_list=augmentation_list,
                 cuda=True,
                 augment_parameters = augment_parameters,
                 model_save_dir='/home/user/DanSpeech-AdvancedMachineLearning/AML Cleaned/model_save_dir/',
                 epochs=epochs,
                 augment_prob_dir = augment_prob_dir
                 
        )


    

    
    
