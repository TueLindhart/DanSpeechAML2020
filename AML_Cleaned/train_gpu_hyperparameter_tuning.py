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
import random
import pandas as pd

    
def f_train_gpu(model_name="NoName",augmentation_list=[],k_folds=None, augment_parameters = None, augment_prob_dir = None, score_ID = None):    
    if __name__ == '__main__':

        # k_folds = None
        
        root_dir = '../Data/Tue_Noise_test_split/'

        factory = DatasetFactory(root_dir)

        if k_folds:

            CV = KFold(k_folds, shuffle=True, random_state=42)

            for train_index, validation_index in CV.split(factory.meta):
                training_set = factory.meta.iloc[train_index]
                validation_set = factory.meta.iloc[validation_index]
                
                if not augment_parameters:
                    augment_parameters = {'speed_perturb'  : [0.9,1.1],
                      'shift_perturb'  : [-50,50],
                      'room_reverb'    : [0, 0.4, 3.0, 0, 5, 3.0, 0, 5, 3.0, 0, 5, 0.5, 0.5, 0.5],
                      'volume_perturb' : [5,30],
                      'add_wn'         : [0.5,1.8],
                      'tempo_perturb'  : [-0.25,0.25],
                      'spec_augment'   : [80, 27, 50, 1, 1]
                     }

                finetune(model_id=model_name,
                         root_dir=root_dir,
                         training_set=training_set,
                         validation_set=validation_set,
                         in_memory=True,
                         stored_model='/home/user/.danspeech/models/DanSpeechPrimary.pth',
                         augment_w_specaug=True,
                         #augmentation_list=["tempo_perturb","room_reverb","volume_perturb","add_wn","shift_perturb","spec_augment"]
                         #augmentation_list = ["tempo_perturb"],
                         augmentation_list=augmentation_list,
                         cuda=True,
                         augment_parameters = augment_parameters,
                         epochs=15,
                         augment_prob_dir = augment_prob_dir,
                         score_ID = score_ID
                )


#         else:
#             # -- example code for training a new model
#             # train_new(model_id=None, train_data_path='danspeech-audiofiles-preprocessor/data/training',
#             #          validation_data_path='danspeech-audiofiles-preprocessor/data/validation', cuda=True)

#             # -- example code for continuation of a new model
#             # continue_training(model_id='DanSpeechPrimary_continued',
#             #                   train_data_path="/home/user/danspeech_training/danspeech-audiofiles-preprocessor/data/tue/training",
#             #                   validation_data_path='/home/user/danspeech_training/danspeech-audiofiles-preprocessor/data/tue/validation',
#             #                   stored_model = '/home/user/.danspeech/models/DanSpeechPrimary_finetuned.pth',
#             #                   cuda=True)
#             #  -- example code of finetuning a model

#             factory.split_file(0.67)
#             finetune(model_id="DanSpeechPrimary_finetuned_200_w_audio_aug_tue",
#                      root_dir=root_dir,
#                      training_set='train.csv',
#                      validation_set='validation.csv',
#                      stored_model='/home/user/.danspeech/models/DanSpeechPrimary.pth',
#                      augment_w_specaug=True,
#                      # augmentation_list = ["room_reverb","volume_perturb","add_wn","shift_perturb","spec_augment"],
#                      # augmentation_list = ["spec_augment"],
#                      augmentation_list=["room_reverb", "volume_perturb", "add_wn", "shift_perturb"],
#                      cuda=True,
#                      augment_parameters = augment_parameters,
#                      epochs = 10
#                      augment_prob_dir = augment_prob_dir
#             )
            
            
            
            
            
            
            
    
# def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

#     log = file if hasattr(file,'write') else sys.stderr
#     traceback.print_stack(file=log)
#     log.write(warnings.formatwarning(message, category, filename, lineno, line))

# warnings.showwarning = warn_with_traceback
    
#augmentation_list=["tempo_perturb","room_reverb","volume_perturb","add_wn","shift_perturb","spec_augment","speed_perturb"]
augmentation_list=["room_reverb","spec_augment"]
k_folds = 3
number_of_iterations = 25
    
#f_train_gpu(model_name = model_name, k_folds = k_folds)    

augment_parameters = {'speed_perturb'  : [0.9,1.1],
                      'shift_perturb'  : [-50,50],
                      'room_reverb'    : [0, 0.4, 3.0, 0, 5, 3.0, 0, 5, 3.0, 0, 5, 0.5, 0.5, 0.5],
                      'volume_perturb' : [5,30],
                      'add_wn'         : [0.5,1.8],
                      'tempo_perturb'  : [-0.25,0.25],
                      'spec_augment'   : [80, 27, 50, 1, 1]
                     }



scores = {}
txt_filename = "Third_HPT_test_all.txt"
f=open(txt_filename, "a+")

import time

start_time = time.time()

for i in range(len(augmentation_list)):
    
    cur_aug = augmentation_list[i] #The current augmentation method
    
    for iteration in range(number_of_iterations): # Do x iterations of random search
        
        augment_prob_dir = {'speed_perturb'  : random.uniform(0,1),
                        'shift_perturb'  : random.uniform(0,1),
                        'room_reverb'    : random.uniform(0,1),
                        'volume_perturb' : random.uniform(0,1),
                        'add_wn'         : random.uniform(0,1),
                        'tempo_perturb'  : random.uniform(0,1),
                        'spec_augment'   : random.uniform(0,1)
                       }
        model_name = "Noisy_hyperparameter_tuning_"
        score_ID = model_name
        vali = False
        
        if cur_aug == 'speed_perturb':
            while not vali:
                augment_parameters[cur_aug][0] = random.uniform(0.7,1.2)
                augment_parameters[cur_aug][1] = random.uniform(0.8,1.3)
                if augment_parameters[cur_aug][0] <= augment_parameters[cur_aug][1]:
                    vali = True
                
        elif cur_aug == 'shift_perturb':
            while not vali:
                augment_parameters[cur_aug][0] = random.uniform(-100,0)
                augment_parameters[cur_aug][1] = random.uniform(0,100)
                if augment_parameters[cur_aug][0] <= augment_parameters[cur_aug][1]:
                    vali = True

        elif cur_aug == 'room_reverb':
            vali1 = False
            vali2 = False
            vali3 = False
            vali4 = False
            # Absorbtion effect (Alpha)
            while not vali1:
                augment_parameters[cur_aug][0] = random.uniform(0,1)
                augment_parameters[cur_aug][1] = random.uniform(0,1)
                if augment_parameters[cur_aug][0] <= augment_parameters[cur_aug][1]:
                    vali1 = True
            # Room length
            while not vali2:
                augment_parameters[cur_aug][2] = random.uniform(3,8)
                augment_parameters[cur_aug][3] = 0
                augment_parameters[cur_aug][4] = 5
                if augment_parameters[cur_aug][3] <= augment_parameters[cur_aug][4]:
                    vali2 = True
            # Room width
            while not vali3:
                augment_parameters[cur_aug][5] = random.uniform(3,8)
                augment_parameters[cur_aug][6] = 0
                augment_parameters[cur_aug][7] = 5
                if augment_parameters[cur_aug][6] <= augment_parameters[cur_aug][7]:
                    vali3 = True
            # Room height. Calculated as (in meters) [3+[6],3+[7]]
            while not vali4:
                augment_parameters[cur_aug][8] = random.uniform(3,8)
                augment_parameters[cur_aug][9] = 0
                augment_parameters[cur_aug][10] = 5
                if augment_parameters[cur_aug][9] <= augment_parameters[cur_aug][10]:
                    vali4 = True
            # Microphone placement x-coordinate
            augment_parameters[cur_aug][10] = 0.5
            # Microphone placement y-coordinate
            augment_parameters[cur_aug][11] = 0.5
            # Microphone placement height (z-coordinate)
            augment_parameters[cur_aug][12] = 0.5
            
            

        elif cur_aug == 'volume_perturb':
            while not vali:
                augment_parameters[cur_aug][0] = random.uniform(2,15)
                augment_parameters[cur_aug][1] = random.uniform(5,45)
                if augment_parameters[cur_aug][0] <= augment_parameters[cur_aug][1]:
                    vali = True

        elif cur_aug == 'add_wn':
            while not vali:
                augment_parameters[cur_aug][0] = random.uniform(0.25,0.75)
                augment_parameters[cur_aug][1] = random.uniform(0.5,2.5)
                if augment_parameters[cur_aug][0] <= augment_parameters[cur_aug][1]:
                    vali = True

        elif cur_aug == 'tempo_perturb':
            while not vali:
                augment_parameters[cur_aug][0] = random.uniform(-0.75,0.25)
                augment_parameters[cur_aug][1] = random.uniform(-0.25,0.75)
                if augment_parameters[cur_aug][0] <= augment_parameters[cur_aug][1]:
                    vali = True

        elif cur_aug == 'spec_augment':
            augment_parameters[cur_aug][0] = random.uniform(50,100)
            augment_parameters[cur_aug][1] = random.uniform(20,40)
            augment_parameters[cur_aug][2] = random.uniform(35,65)
            augment_parameters[cur_aug][3] = 1
            augment_parameters[cur_aug][4] = 1 

        aug_method = []
        aug_method.append(cur_aug)
        cur_iter = cur_aug + "_Rand" + str(round(augment_prob_dir[cur_aug],2))
        model_name = model_name + augmentation_list[i]
        print("[%d] Now training %s with hyperparameters: " %(iteration,model_name))
        for k in range(len(augment_parameters[cur_aug])):
            score_ID = score_ID + "_" + str(round(augment_parameters[cur_aug][k],2))
            cur_iter = cur_iter + "_" + str(round(augment_parameters[cur_aug][k],2))
            
            print("\t%s" %(str(round(augment_parameters[cur_aug][k],2))))

        # Train the model
        f_train_gpu(model_name = model_name, augmentation_list = aug_method, k_folds = k_folds, 
                    augment_parameters=augment_parameters, augment_prob_dir = augment_prob_dir,
                    score_ID = score_ID)
    
        
        path = os.path.join(os.getcwd(),"Scores","allScores.csv")
        df = pd.read_csv(path,sep=";")
        means = df.groupby("score_ID").mean()[["WER","CER","Best_epoch"]]
        this_score = means.loc[score_ID]
        
        scores[cur_iter] = [this_score[0],this_score[1],this_score[2]]
        
        f.write("%s %s %s %s\n" %(cur_iter,str(this_score[0]),str(this_score[1]),str(this_score[2])))
        
print(scores)  
print("Done! Elapsed time: %.2f minutes" %( (time.time()-start_time)/60 ))
f.write("Elapsed time: %.2f minutes" %( (time.time()-start_time)/60 ))
f.close()
