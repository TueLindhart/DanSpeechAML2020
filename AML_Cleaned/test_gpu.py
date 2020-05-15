#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 16:57:13 2020

@author: user
"""

import os
import sys
import torch
from deepspeech.test import test_model

if torch.cuda.is_available():
    print("Success")
    print(torch.cuda.device_count())
    

def test_GPU(model_path, data_path):    
    print("Greedy:")
    #greddy_WER, greedy_CER = test_model(model_path = model_path,
    test_model(model_path = model_path,
               data_path = data_path,
                decoder="greedy", 
                cuda=True, 
                batch_size=40, 
                num_workers=4,
                lm_path=None, 
                alpha=1.3, 
                beta=0.4, 
                cutoff_top_n=40, 
                cutoff_prob=1.0, 
                beam_width=64,
                lm_workers=4, 
                output_path="/home/user/DanSpeech-AdvancedMachineLearning/Tests", 
                verbose=False)

    print("\n")
    print("3-gram language model:")
    #gram3_WER, gram3_CER = test_model(model_path = model_path,
    test_model(model_path = model_path,
               data_path = data_path,
               decoder="beam", 
               cuda=True, 
               batch_size=40, 
               num_workers=4,
               lm_path="/home/user/.danspeech/lms/dsl_3gram.klm", 
               alpha=1.3, 
               beta=0.4, 
               cutoff_top_n=40, 
               cutoff_prob=1.0,
               beam_width=64,
               lm_workers=4, 
               output_path="/home/user/DanSpeech-AdvancedMachineLearning/Tests", 
               verbose=False)


    print("\n")
    print("5-gram language model:")
    #gram5_WER, gram5_CER = test_model(model_path = model_path,
    test_model(model_path = model_path,
               data_path = data_path,
               decoder="beam", 
               cuda=True, 
               batch_size=40, 
               num_workers=4,
               lm_path="/home/user/.danspeech/lms/dsl_5gram.klm", 
               alpha=1.3, 
               beta=0.4, 
               cutoff_top_n=40, 
               cutoff_prob=1.0, 
               beam_width=64,
               lm_workers=4, 
               output_path="/home/user/DanSpeech-AdvancedMachineLearning/Tests", 
               verbose=False)
    
    #results = {"Greedy": {"WER": greddy_WER,
    #                      "CER": greedy_CER},
    #           "3-gram": {"WER": gram3_WER,
    #                      "CER": gram3_CER},
    #           "5-gram": {"WER": gram5_WER,
    #                      "CER": gram5_CER}
    return #results