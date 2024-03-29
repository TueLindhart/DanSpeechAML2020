import os
import time
from tqdm import tqdm
import warnings
import json
from datetime import datetime

import torch
# -- warpctc bindings for pytorch can be found here: https://github.com/SeanNaren/warp-ctc
from warpctc_pytorch import CTCLoss

from audio.datasets import BatchDataLoader, DanSpeechDataset
from audio.dataset_factory import DatasetFactory
from audio.parsers import SpectrogramAudioParser, AugmenterAudioParser
from audio.augmentation import DanSpeechAugmenter
from danspeech.deepspeech.model import DeepSpeech, supported_rnns
from danspeech.deepspeech.decoder import GreedyDecoder
from deepspeech.training_utils import TensorBoardLogger, AverageMeter, reduce_tensor, sum_tensor, get_default_audio_config, serialize, loadLogger, saveScores, csvSaver
from danspeech.errors.training_errors import ArgumentMissingForOption


class NoModelSaveDirSpecified(Warning):
    pass


class NoLoggingDirSpecified(Warning):
    pass


class NoModelNameSpecified(Warning):
    pass


class InfiniteLossReturned(Warning):
    pass


def _train_model(model_id=None, training_set=None, validation_set=None, root_dir=None,
                 in_memory=False, epochs=20, stored_model=None,
                 model_save_dir=None, tensorboard_log_dir=None, augmented_training=False, batch_size=8,
                 num_workers=6, cuda=False, lr=3e-4, momentum=0.9, weight_decay=1e-5, max_norm=400,
                 context=20, continue_train=False, finetune=False, train_new=False, num_freeze_layers=None,
                 rnn_type='gru', conv_layers=2, rnn_hidden_layers=5, rnn_hidden_size=800,
                 bidirectional=True, distributed=False, gpu_rank=None, dist_backend='nccl', rank=0,
                 dist_url='tcp://127.0.0.1:1550', world_size=1, 
                 augment_w_specaug = False,
                 augmentation_list = [],
                 augment_parameters = None,
                 augment_prob_dir = None,
                 score_ID = None):

    # Load scores
    scoresDict = loadLogger()
    
    # Add cross-val fold 
    counter = 1
    if score_ID != None:
        ID = score_ID
    else:
        ID = model_id
        
    while ID in list(scoresDict.keys()):
        if ID[-2] != "_":
            ID = ID + "_" + str(counter)
        else:
            ID = ID[:-1] + str(counter)
        counter += 1
    
    # Add Time
    currTime = datetime.now()
    currTime = time.strftime("%d/%m/%Y, %H:%M:%S")
    
    # Set values
    scoresDict[ID] = {"Time": currTime,
                      "Augmentations": augmentation_list,
                      "Avg_WER": 0,
                      "Avg_CER": 0}
    
    # -- set training device
    main_proc = True
    device = torch.device("cuda" if cuda else "cpu")

    # -- prepare directories for storage and logging.
    if not model_save_dir:
        warnings.warn("You did not specify a directory for saving the trained model.\n"
                      "Defaulting to ~/.danspeech/custom/ directory.", NoModelSaveDirSpecified)

        model_save_dir = os.path.join(os.path.expanduser('~'), '.danspeech/models/')

    os.makedirs(model_save_dir, exist_ok=True)

    if not model_id:
        warnings.warn("You did not specify a name for the trained model.\n"
                      "Defaulting to danish_speaking_panda.pth", NoModelNameSpecified)

        model_id = "danish_speaking_panda"

    if main_proc and tensorboard_log_dir:
        logging_process = True
        tensorboard_logger = TensorBoardLogger(model_id, tensorboard_log_dir)
    else:
        logging_process = False
        warnings.warn(
            "You did not specify a directory for logging training process. Training process will not be logged.",
            NoLoggingDirSpecified)

    # -- handle distributed processing
    if distributed:
        import torch.distributed as dist
        from torch.utils.data.distributed import DistributedSampler
        from apex.parallel import DistributedDataParallel

        if gpu_rank:
            torch.cuda.set_device(int(gpu_rank))

        dist.init_process_group(backend=dist_backend, init_method=dist_url,
                                world_size=world_size, rank=rank)

    # -- initialize training metrics
    loss_results = torch.Tensor(epochs)
    cer_results = torch.Tensor(epochs)
    wer_results = torch.Tensor(epochs)

    # -- initialize helper variables
    avg_loss = 0
    start_epoch = 0
    start_iter = 0

    # -- load and initialize model metrics based on wrapper function
    #if train_new:
    #    with open(os.path.dirname(os.path.realpath(__file__)) + '/labels.json', "r", encoding="utf-8") as label_file:
    #        labels = str(''.join(json.load(label_file)))
#
    #    # -- changing the default audio config is highly experimental, make changes with care and expect vastly
    #    # -- different results compared to baseline
    #    audio_conf = get_default_audio_config()
#
    #    rnn_type = rnn_type.lower()
    #    conv_layers = conv_layers
    #    assert rnn_type in ["lstm", "rnn", "gru"], "rnn_type should be either lstm, rnn or gru"
    #    assert conv_layers in [1, 2, 3], "conv_layers must be set to either 1, 2 or 3"
    #    model = DeepSpeech(model_name=model_id,
    #                       conv_layers=conv_layers,
    #                       rnn_hidden_size=rnn_hidden_size,
    #                       rnn_layers=rnn_hidden_layers,
    #                       labels=labels,
    #                       rnn_type=supported_rnns.get(rnn_type),
    #                       audio_conf=audio_conf,
    #                       bidirectional=bidirectional,
    #                       streaming_inference_model=False,  # -- streaming inference should always be disabled during training
    #                       context=context)
    #    parameters = model.parameters()
    #    optimizer = torch.optim.SGD(parameters, lr=lr,
    #                                momentum=momentum, nesterov=True, weight_decay=1e-5)

    if finetune:
        if not stored_model:
            raise ArgumentMissingForOption("If you want to finetune, please provide the absolute path"
                                           "to a trained pytorch model object as the stored_model argument")
        else:
            print("Loading checkpoint model %s" % stored_model)
            package = torch.load(stored_model, map_location=lambda storage, loc: storage)
            model = DeepSpeech.load_model_package(package)

            if num_freeze_layers:
                # -- freezing layers might result in unexpected results, use with cation
                print("Freezing of layers initiated")
                model.freeze_layers(num_freeze_layers)

            parameters = model.parameters()
            optimizer = torch.optim.SGD(parameters, lr=lr,
                                        momentum=momentum, nesterov=True, weight_decay=1e-5)

            if logging_process:
                tensorboard_logger.load_previous_values(start_epoch, package)

    #if continue_train:
    #    # -- continue_training wrapper
    #    if not stored_model:
    #        raise ArgumentMissingForOption("If you want to continue training, please support a package with previous"
    #                                       "training information or use the finetune option instead")
    #    else:
    #        print("Loading checkpoint model %s" % stored_model)
    #        package = torch.load(stored_model, map_location=lambda storage, loc: storage)
    #        model = DeepSpeech.load_model_package(package)
    #        # -- load stored training information
    #        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
    #                                    nesterov=True, weight_decay=1e-5)
    #        optim_state = package['optim_dict']
    #        optimizer.load_state_dict(optim_state)
    #        start_epoch = int(package['epoch']) + 1  # -- Index start at 0 for training
#
    #        print("Last successfully trained Epoch: {0}".format(start_epoch))
#
    #        start_epoch += 1
    #        start_iter = 0
#
    #        avg_loss = int(package.get('avg_loss', 0))
    #        loss_results_ = package['loss_results']
    #        cer_results_ = package['cer_results']
    #        wer_results_ = package['wer_results']
#
    #        # ToDo: Make depend on the epoch from the package
    #        previous_epochs = loss_results_.size()[0]
    #        print("Previously set to run for: {0} epochs".format(previous_epochs))
#
    #        loss_results[0:previous_epochs] = loss_results_
    #        wer_results[0:previous_epochs] = cer_results_
    #        cer_results[0:previous_epochs] = wer_results_
#
    #        if logging_process:
    #            tensorboard_logger.load_previous_values(start_epoch, package)


#     if augment_w_specaug:
    training_parser = AugmenterAudioParser(audio_config=model.audio_conf,augmentation_list=augmentation_list,
                                           augment_args=augment_parameters,augment_prob_dir=augment_prob_dir)
    validation_parser = SpectrogramAudioParser(audio_config=model.audio_conf, data_augmenter=None)
    
    
#     else:

#         # -- initialize DanSpeech augmenter
#         if augmented_training:
#             print("Augmentations started")
#             #augmenter = DanSpeechAugmenter(sampling_rate=model.audio_conf["sampling_rate"])
#             augmenter = DanSpeechAugmenter(sampling_rate=model.audio_conf["sample_rate"])
            
#         else:
#             augmenter = None
    
#         # -- initialize audio parser and dataset
#         # -- audio parsers
#         training_parser = SpectrogramAudioParser(audio_config=model.audio_conf, data_augmenter=augmenter)
#         validation_parser = SpectrogramAudioParser(audio_config=model.audio_conf, data_augmenter=None)
    
    # -- instantiate data-sets
    training_set = DanSpeechDataset(root_dir, training_set, labels=model.labels, audio_parser=training_parser, in_memory=in_memory)
    validation_set = DanSpeechDataset(root_dir, validation_set, labels=model.labels, audio_parser=validation_parser, in_memory=in_memory)
    
    # -- Tue: extracting meta data for validation set such as file names
    meta = validation_set.meta

    print("")
    # -- initialize batch loaders
    if not distributed:
        # -- initialize batch loaders for single GPU or CPU training
        train_batch_loader = BatchDataLoader(training_set, batch_size=batch_size, num_workers=num_workers,
                                             shuffle=True, pin_memory=True)
        validation_batch_loader = BatchDataLoader(validation_set, batch_size=batch_size, num_workers=num_workers,
                                                  shuffle=False)
    else:
        # -- initialize batch loaders for distributed training on multiple GPUs
        train_sampler = DistributedSampler(training_set, num_replicas=world_size, rank=rank)
        train_batch_loader = BatchDataLoader(training_set, batch_size=batch_size,
                                             num_workers=num_workers,
                                             sampler=train_sampler,
                                             pin_memory=True)

        validation_sampler = DistributedSampler(validation_set, num_replicas=world_size, rank=rank)
        validation_batch_loader = BatchDataLoader(validation_set, batch_size=batch_size,
                                                  num_workers=num_workers,
                                                  sampler=validation_sampler)

        model = DistributedDataParallel(model)

    decoder = GreedyDecoder(model.labels)
    criterion = CTCLoss()
    model = model.to(device)
    best_wer = None

    # -- verbatim training outputs during progress
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    print(model)
    print("Initializations complete, starting training pass on model: %s \n" % model_id)
    print("Number of parameters: %d \n" % DeepSpeech.get_param_size(model))
    
    try:
        for epoch in range(start_epoch, epochs):
            if distributed and epoch != 0:
                # -- distributed sampling, keep epochs on all GPUs
                train_sampler.set_epoch(epoch)

            print('started training epoch %d' % (epoch + 1))
            model.train()

            # -- timings per epoch
            end = time.time()
            start_epoch_time = time.time()
            num_updates = len(train_batch_loader)

            # -- per epoch training loop, iterate over all mini-batches in the training set
            for i, (data) in enumerate(train_batch_loader, start=start_iter):
                if i == num_updates:
                    break

                # -- grab and prepare a sample for a training pass
                inputs, targets, input_percentages, target_sizes = data
                input_sizes = input_percentages.mul_(int(inputs.size(3))).int()

                # -- measure data load times, this gives an indication on the number of workers required for latency
                # -- free training.
                data_time.update(time.time() - end)

                # -- parse data and perform a training pass
                inputs = inputs.to(device)

                # -- compute the CTC-loss and average over mini-batch
                out, output_sizes = model(inputs, input_sizes)
                out = out.transpose(0, 1)
                float_out = out.float()
                loss = criterion(float_out, targets, output_sizes, target_sizes).to(device)
                loss = loss / inputs.size(0)

                # -- check for diverging losses
                if distributed:
                    loss_value = reduce_tensor(loss, world_size).item()
                else:
                    loss_value = loss.item()

                if loss_value == float("inf") or loss_value == -float("inf"):
                    warnings.warn("received an inf loss, setting loss value to 0", InfiniteLossReturned)
                    loss_value = 0

                # -- update average loss, and loss tensor
                avg_loss += loss_value
                losses.update(loss_value, inputs.size(0))

                # -- compute gradients and back-propagate errors
                optimizer.zero_grad()
                loss.backward()

                # -- avoid exploding gradients by clip_grad_norm, defaults to 400
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

                # -- stochastic gradient descent step
                optimizer.step()

                # -- measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                #print('Epoch: [{0}/{1}][{2}/{3}]\t'
                #      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                #      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                #      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                #    (epoch + 1), (epochs), (i + 1), len(train_batch_loader), batch_time=batch_time,
                #    data_time=data_time, loss=losses))

                del loss, out, float_out

            # -- report epoch summaries and prepare validation run
            avg_loss /= len(train_batch_loader)
            loss_results[epoch] = avg_loss
            epoch_time = time.time() - start_epoch_time
            print('Training Summary Epoch: [{0}]\t'
                  'Time taken (s): {epoch_time:.0f}\t'
                  'Average Loss {loss:.3f}\t'.format(epoch + 1, epoch_time=epoch_time, loss=avg_loss))

            # -- prepare validation specific parameters, and set model ready for evaluation
            total_cer, total_wer = 0, 0
            model.eval() 
            # -- Tue: transcriptScore here so it only saves the current epoch
            transcriptScore = []
            with torch.no_grad():
                for i, (data) in tqdm(enumerate(validation_batch_loader), total=len(validation_batch_loader)):
                    inputs, targets, input_percentages, target_sizes = data
                    input_sizes = input_percentages.mul_(int(inputs.size(3))).int()

                    # -- unflatten targets
                    split_targets = []
                    offset = 0
                    targets = targets.numpy()
                    for size in target_sizes:
                        split_targets.append(targets[offset:offset + size])
                        offset += size

                    inputs = inputs.to(device)
                    out, output_sizes = model(inputs, input_sizes)
                    decoded_output, _ = decoder.decode(out, output_sizes)
                    target_strings = decoder.convert_to_strings(split_targets)
                    

                    # -- compute accuracy metrics
                    wer, cer = 0, 0
                    
                    for x in range(len(target_strings)):
         
                        transcript, reference = decoded_output[x][0], target_strings[x][0]
                        wer = decoder.wer(transcript, reference) / float(len(reference.split()))
                        cer = decoder.cer(transcript, reference) / float(len(reference))
                        
                        total_wer += wer #Tue: - Changed so saving of wer per audio file possible. 
                        total_cer += cer #Tue: - Changed so saving of wer per audio file possible
                        
                        #filename,_ = meta[n] -- Does not choose the correct filename

                        transcriptScore.append([currTime,score_ID, reference, wer, cer, transcript, augmentation_list,
                                                epoch+1,epochs])
                        

                    #total_wer += wer --Tue: - so saving of wer per audio file possible. 
                    #total_cer += cer --Tue: - so saving of wer per audio file possible. 
                    del out

            if distributed:
                # -- sums tensor across all devices if distributed training is enabled
                total_wer_tensor = torch.tensor(total_wer).to(device)
                total_wer_tensor = sum_tensor(total_wer_tensor)
                total_wer = total_wer_tensor.item()

                total_cer_tensor = torch.tensor(total_cer).to(device)
                total_cer_tensor = sum_tensor(total_cer_tensor)
                total_cer = total_cer_tensor.item()

                del total_wer_tensor, total_cer_tensor

            # -- compute average metrics for the validation pass
            avg_wer_epoch = (total_wer / len(validation_batch_loader.dataset)) * 100
            avg_cer_epoch = (total_cer / len(validation_batch_loader.dataset)) * 100            
            
            # -- append metrics for logging
            loss_results[epoch], wer_results[epoch], cer_results[epoch] = avg_loss, avg_wer_epoch, avg_cer_epoch
            
            # -- Johan: Logging
            
            if epoch > epoch-5:
                scoresDict[ID]['Avg_WER'] += avg_wer_epoch
                scoresDict[ID]['Avg_CER'] += avg_cer_epoch
            else:
                pass
            #if epoch > 0:
            #    scoresDict[model_id]['Avg_WER'] = scoresDict[model_id]['Avg_WER']/(epoch+1)
            #    scoresDict[model_id]['Avg_CER'] = scoresDict[model_id]['Avg_CER']/(epoch+1)
            #else:
            #    pass
            #saveScores(scoresDict)
            
            # -- log metrics for tensorboard
            if logging_process:
                logging_values = {
                    "loss_results": loss_results,
                    "wer": avg_wer_epoch,
                    "cer": avg_cer_epoch
                }
                tensorboard_logger.update(epoch, logging_values)

            # -- print validation metrics summary
            print('Validation Summary Epoch: [{0}]\t'
                  'Average WER {wer:.3f}\t'
                  'Average CER {cer:.3f}\t'.format(epoch + 1, wer=avg_wer_epoch, cer=avg_cer_epoch))

            # -- save model if it has the highest recorded performance on validation.
            #OBS! changed from best_wer > wer to best_wer > avg_wer_epoch - Tue
            if main_proc and (best_wer is None) or (best_wer > avg_wer_epoch):
                model_path = model_save_dir + model_id + '.pth'
                best_transcript = transcriptScore.copy() # -Tue: Saving best performance to be saved in .csv
                

                # -- check if the model is uni or bidirectional, and set streaming model accordingly
                if not bidirectional:
                    streaming_inference_model = True
                else:
                    streaming_inference_model = False
                print("Found better validated model, saving to %s" % model_path)
                torch.save(serialize(model, optimizer=optimizer, epoch=epoch, loss_results=loss_results,
                                     wer_results=wer_results, cer_results=cer_results,
                                     distributed=distributed, streaming_model=streaming_inference_model,
                                     context=context), model_path)

                best_wer = avg_wer_epoch #OBS! changed from wer to avg_wer_epoch - Tue
                avg_loss = 0
                
            

            # -- reset start iteration for next epoch
            start_iter = 0
        
        if epoch > epochs-6:
            scoresDict[ID]['Avg_WER'] = scoresDict[ID]['Avg_WER']/(epochs-5)
            scoresDict[ID]['Avg_CER'] = scoresDict[ID]['Avg_CER']/(epochs-5)
            saveScores(scoresDict)
            
        if epoch == (epochs-1): # when last epoch is run the scores will be saved or with keyboardinterrupt
            csvSaver(best_transcript)

    except KeyboardInterrupt:
        print('Successfully exited training and stopped all processes.')
        csvSaver(best_transcript)

        
#def train_new(model_id, training_set=None, validation_set=None, root_dir=None, in_memory=False,
#              conv_layers=2, rnn_type='gru', rnn_hidden_layers=5,
#              rnn_hidden_size=800, bidirectional=True, epochs=20, model_save_dir=None,
#              tensorboard_log_dir=None, **args):
#
#    _train_model(model_id, training_set=training_set, validation_set=validation_set, root_dir=root_dir,
#                 in_memory=in_memory, conv_layers=conv_layers, rnn_type=rnn_type,
#                 rnn_hidden_layers=rnn_hidden_layers, rnn_hidden_size=rnn_hidden_size, bidirectional=bidirectional,
#                 model_save_dir=model_save_dir, tensorboard_log_dir=tensorboard_log_dir, train_new=True,
#                 epochs=epochs, augmented_training=False, **args)


def finetune(model_id, training_set=None, validation_set=None, root_dir=None, in_memory=False, epochs=15, stored_model=None,
             model_save_dir=None,
             tensorboard_log_dir=None, num_freeze_layers=None,augment_w_specaug=True,
             augmentation_list = ["tempo_perturb","room_reverb","volume_perturb","add_wn","shift_perturb","spec_augment"],augment_parameters=None,
             lr=(3e-4)/2, score_ID = None,
             **args):

    _train_model(model_id, training_set=training_set, validation_set=validation_set, root_dir=root_dir,
                 in_memory=in_memory, epochs=epochs, stored_model=stored_model,
                 model_save_dir=model_save_dir, tensorboard_log_dir=tensorboard_log_dir, finetune=True,
                 num_freeze_layers=num_freeze_layers,augmented_training=False,
                 augment_w_specaug = augment_w_specaug, 
                 augmentation_list = augmentation_list,
                 augment_parameters = augment_parameters,
                 lr=lr, score_ID = score_ID,
                 **args)

#def continue_training(model_id, training_set=None, validation_set=None, root_dir=None, in_memory=False, epochs=20,
#                      stored_model=None, model_save_dir=None, tensorboard_log_dir=None, **args):
#
#    _train_model(model_id, training_set=training_set, validation_set=validation_set, root_dir=root_dir,
#                 in_memory=in_memory, epochs=epochs, stored_model=stored_model,
#                 model_save_dir=model_save_dir, tensorboard_log_dir=tensorboard_log_dir, continue_train=True,
#                 augmented_training=True, **args)
#