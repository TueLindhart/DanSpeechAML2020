import torch
import numpy as np
from tqdm import tqdm
from danspeech.deepspeech.decoder import GreedyDecoder
from danspeech.deepspeech.model import DeepSpeech
from audio.parsers import SpectrogramAudioParser
from audio.datasets import DanSpeechDataset, BatchDataLoader
from deepspeech.training_utils import csvSaverTest, loadCSVTest


def test_model(model_path, data_path, decoder="greedy", cuda=False, batch_size=96, num_workers=4,
               lm_path=None, alpha=1.3, beta=0.4, cutoff_top_n=40, cutoff_prob=1.0, beam_width=64,
               lm_workers=4, output_path=None, verbose=False):
    torch.set_grad_enabled(False)
    model = DeepSpeech.load_model(model_path)
    device = torch.device("cuda" if cuda else "cpu")
    model = model.to(device)
    model.eval()
    
    # -- Save name of model name and language model to when scores are saved 
    model_name = model_path.split("/")[-1]
    data_name = data_path.split("/")[-1]
    
    if lm_path is None: 
        lm = "None"
    else:
        lm = lm_path.split("/")[-1]
        
    print(model_name)

    # -- Tue: Implemented to make beam search possible
    if decoder == "beam":
        from danspeech.deepspeech.decoder import BeamCTCDecoder
        
        decode_type = decoder
        decoder = BeamCTCDecoder(model.labels, lm_path=lm_path, alpha=alpha, beta=beta,
                                 cutoff_top_n=cutoff_top_n, cutoff_prob=cutoff_prob,
                                 beam_width=beam_width, num_processes=lm_workers)
        greedy_decoder = GreedyDecoder(model.labels, blank_index=model.labels.index('_'))
        
    elif decoder == "greedy":
        
        decode_type = decoder
        decoder = GreedyDecoder(model.labels, blank_index=model.labels.index('_'))
    else:
        raise AttributeError("please specify a valid decoder, DanSpeech currently supports [greedy, beam]")

    test_parser = SpectrogramAudioParser(audio_config=model.audio_conf, data_augmenter=None)
    test_dataset = DanSpeechDataset(data_path, 'test.csv', labels=model.labels, audio_parser=test_parser)
    test_batch_loader = BatchDataLoader(test_dataset, batch_size=batch_size,
                                        num_workers=num_workers, shuffle=False)

    total_cer, total_wer, num_tokens, num_chars = 0, 0, 0, 0
    output_data = []
    testScore = [] # -- Tue: Creating aémpty list to save each score
    for i, (data) in tqdm(enumerate(test_batch_loader), total=len(test_batch_loader)):
        inputs, targets, input_percentages, target_sizes = data
        input_sizes = input_percentages.mul_(int(inputs.size(3))).int()

        split_targets = []
        offset = 0
        targets = targets.numpy()
        for size in target_sizes:
            split_targets.append(targets[offset:offset + size])
            offset += size

        inputs = inputs.to(device)
        out, output_sizes = model(inputs, input_sizes)

        if decoder is None:
            output_data.append((out.numpy(), output_sizes.numpy()))
            continue
        

        decoded_output, _  = decoder.decode(out.data, output_sizes.data)
        
        # -- Tue: Implemented to make beam search possible
        if decode_type == "greedy":
            target_strings = decoder.convert_to_strings(split_targets) 
        elif decode_type == "beam":
            target_strings = greedy_decoder.convert_to_strings(split_targets)
             
            
        for x in range(len(target_strings)):
            transcript, reference = decoded_output[x][0], target_strings[x][0]
            wer_inst = decoder.wer(transcript, reference)
            cer_inst = decoder.cer(transcript, reference)
            total_wer += wer_inst
            total_cer += cer_inst
            num_tokens += len(reference.split())
            num_chars += len(reference)
            
            # -- Tue: Appending save score to make statistics possible
            testScore.append([data_name,decode_type,lm,model_name, reference.lower(), float(wer_inst) / len(reference.split()),
                          float(cer_inst) / len(reference),transcript.lower()])
            
            if verbose:
                print("Ref:", reference.lower())
                print("Hyp:", transcript.lower())
                print("WER:", float(wer_inst) / len(reference.split()), "CER:", float(cer_inst) / len(reference), "\n")

    if decoder is not None:
        wer = float(total_wer) / num_tokens
        cer = float(total_cer) / num_chars

        print('Test Summary \t'
              'Average WER {wer:.3f}\t'
              'Average CER {cer:.3f}\t'.format(wer=wer * 100, cer=cer * 100))
        
        #-- Tue: Saving score for testing 
        csvSaverTest(testScore)
        
        #return wer, cer
    elif output_path is not None:
        np.save(output_path, output_data)
    else:
        pass
    
    return