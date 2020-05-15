import os

import pandas as pd

import torch
from danspeech.audio.resources import load_audio_wavPCM
from torch.utils.data import Dataset, DataLoader


class DanSpeechDataset(Dataset):
    """
    Specifies a generator class for speech data
    located in a root directory. Speech data must be
    in .wav format and the root directory must include
    a .csv file with a list of filenames.

    Samples can be obtained my the __getitem__ method.
    Samples can be augmented by specifying a list of
    transform classes. DanSpeech offers a variety of
    premade transforms.
    """

    def __init__(self, root_dir, csv_file, labels, audio_parser, in_memory=False):

        print("Loading dataset...")
        self.audio_parser = audio_parser
        self.root_dir = root_dir
        self.labels_map = dict([(labels[i], i) for i in range(len(labels))])

        # ToDO: Should not rely on pandas
        if in_memory:
            meta = csv_file
        else:
            meta = pd.read_csv(os.path.join(self.root_dir, 'sets', csv_file), encoding="utf-8")

        meta = list(zip(meta["file"].values, meta["trans"].values))

        # Check that all files exist
        files_not_found = False

        new_meta = []
        for f, trans in meta:
            if not os.path.isfile(os.path.join(self.root_dir, f)):
                files_not_found = True
            else:
                new_meta.append((f, trans))

        if files_not_found:
            print("Not all audio files in the found csv file were found.")

        keys = list(range(len(new_meta)))
        self.meta = dict(zip(keys, new_meta))

        self.size = len(self.meta)
        print("Length of dataset: {0}".format(self.size))

    def __len__(self):
        return self.size

    def path_gen(self, f):
        return os.path.join(self.root_dir, f)

    def __getitem__(self, idx):
        # ToDo: Consider rewriting load audio to use the SpeechFile audio loading setup and benchmark
        f, trans = self.meta[idx]

        recording = load_audio_wavPCM(path=self.path_gen(f))
        recording = self.audio_parser.parse_audio(recording)

        trans = [self.labels_map.get(c) for c in trans]

        return recording, trans


def _collate_fn(batch):
    def func(p):
        return p[0].size(1)

    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
    longest_sample = max(batch, key=func)[0]
    freq_size = longest_sample.size(0)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)
    inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)
    input_percentages = torch.FloatTensor(minibatch_size)
    target_sizes = torch.IntTensor(minibatch_size)
    targets = []
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(1)
        inputs[x][0].narrow(1, 0, seq_length).copy_(tensor)
        input_percentages[x] = seq_length / float(max_seqlength)
        target_sizes[x] = len(target)
        targets.extend(target)

    targets = torch.IntTensor(targets)
    return inputs, targets, input_percentages, target_sizes


class BatchDataLoader(DataLoader):
    def __init__(self, dataset, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(BatchDataLoader, self).__init__(dataset, *args, **kwargs)
        self.collate_fn = _collate_fn
