from abc import ABC, abstractmethod

import torch

import scipy
import numpy as np
import librosa
import random
import pyroomacoustics as pra

windows = {'hamming': scipy.signal.hamming, 'hann': scipy.signal.hann, 'blackman': scipy.signal.blackman,
           'bartlett': scipy.signal.bartlett}


class AudioParser(ABC):
    """
    Abstract class for AudioParsers
    """

    def __init__(self, audio_config=None):
        self.audio_config = audio_config
        if not self.audio_config:
            self.audio_config = {}

        # Defaulting if no audio_config
        self.normalize = self.audio_config.get("normalize", True)
        self.sampling_rate = self.audio_config.get("sampling_rate", 16000)
        window = self.audio_config.get("window", "hamming")
        self.window = windows[window]

        self.window_stride = self.audio_config.get("window_stride", 0.01)
        self.window_size = self.audio_config.get("window_size", 0.02)

    @abstractmethod
    def parse_audio(self, recording):
        pass


class SpectrogramAudioParser(AudioParser):

    def __init__(self, audio_config=None, data_augmenter=None):
        # inits all audio configs
        super(SpectrogramAudioParser, self).__init__(audio_config)

        self.n_fft = int(self.sampling_rate * self.window_size)
        self.hop_length = int(self.sampling_rate * self.window_stride)

        self.data_augmenter = data_augmenter

    def parse_audio(self, recording):

        if self.data_augmenter:
            recording = self.data_augmenter.augment(recording)

        # STFT
        D = librosa.stft(recording, n_fft=self.n_fft, hop_length=self.hop_length,
                         win_length=self.n_fft, window=self.window)

        spect, phase = librosa.magphase(D)
        # S = log(S+1)
        spect = np.log1p(spect)
        spect = torch.FloatTensor(spect)
        if self.normalize:
            mean = spect.mean()
            std = spect.std()
            spect.add_(-mean)
            spect.div_(std)

        return spect
    
class AugmenterAudioParser(AudioParser):
    """
    AudioParser to augment data randomly before feeding to the network.
    """

    def __init__(self, audio_config=None,augmentation_list=None, augment_args=None,augment_prob_dir=None):
        super(AugmenterAudioParser, self).__init__(audio_config)
        self.n_fft = int(self.sampling_rate * self.window_size)
        self.hop_length = int(self.sampling_rate * self.window_stride)

        if augmentation_list:
            self.augmentations_list = augmentation_list
        else:
            #self.augmentations_list = [
             #   "speed_perturb", "room_reverb", "volume_perturb",
              #  "add_wn", "shift_perturb", "spec_augment"]
            self.augmentations_list = [] # -- Tue: Changed so when list is empty no augmentations are made
            
        if augment_args:
            self.augment_args = augment_args
        else:
            self.augment_args = {'speed_perturb' : [0.9,1.1],
                                'shift_perturb' : [-50,50],
                                'room_reverb' : [0, 0.4, 2, 12, 2, 6, -0.5, 0.5, 0.5, 0.5, 0.5],
                                'volume_perturb' : [5,30],
                                'add_wn' : [0.5,1.8],
                                'tempo_perturb' : [-0.25,0.25],
                                'spec_augment' : [80, 27, 50, 1, 1]}
            
        self.augment_prob_dir = augment_prob_dir
          

    def parse_audio(self, recording):
        scheme = self.choose_augmentation_scheme(self.augmentations_list,self.augment_prob_dir)

        if len(scheme) > 0:
            for augmentation in scheme:
                if augmentation != "spec_augment":
                    augmentor = getattr(self, augmentation)
                    recording = augmentor(recording)

        D = librosa.stft(recording, n_fft=self.n_fft,
                         hop_length=self.hop_length,
                         win_length=self.n_fft,
                         window=self.window)

        spect, phase = librosa.magphase(D)
        spect = np.log1p(spect)
        spect = torch.FloatTensor(spect)

        if self.normalize:
            mean = spect.mean()
            std = spect.std()
            spect.add_(-mean)
            spect.div_(std)

        if "spec_augment" in self.augmentations_list:
            spect = self.spec_augment(spect)

        return spect

    def speed_perturb(self,recording, *args): #self added so sampling_rate is correct
        """
        Select up/down-sampling randomly between 90% and 110% of original sample rate

        :param recording: Recording to be augmented
        :param sampling_rate: Sampling rate of the recording
        :return: Augmented recording
        """
        arguments = self.augment_args['speed_perturb']
        
        rand_low = arguments[0]
        rand_high = arguments[1]
        
        new_sample_rate = self.sampling_rate * random.choice([rand_low,rand_high]) # again the sampling rate should simply be a number
        return librosa.core.resample(recording, self.sampling_rate, new_sample_rate)


    def shift_perturb(self,recording, *args):#self added so sampling_rate is correct
        """
        Shifts the audio recording randomly in time.

        :param recording: Recording to be augmented
        :param sampling_rate: Sampling rate of the recording
        :return: Augmented recording
        """
        
        arguments = self.augment_args['shift_perturb']
        rand_low = arguments[0]
        rand_high = arguments[1]
        
        shift_ms = np.random.randint(low=rand_low, high=rand_high)
        shift_samples = int(shift_ms * self.sampling_rate / 1000)

        if shift_samples > 0:
            # time advance
            recording[:-shift_samples] = recording[shift_samples:]
            recording[-shift_samples:] = 0
        elif shift_samples < 0:
            # time delay
            recording[-shift_samples:] = recording[:shift_samples]
            recording[:-shift_samples] = 0
        return recording


    def room_reverb(self,recording, *args):
                    #self added so sampling_rate is correct
        """
        Perturb signal with room reverberations in a randomly generated shoebox room.

        :param recording: Recording to be augmented
        :param sampling_rate: Sampling rate of the recording
        :return: Augmented recording
        """
        #Prepare arguments
        arguments = self.augment_args['room_reverb']
        alpha_low = arguments[0]
        alpha_high = arguments[1]
        
        roomlen_mean = arguments[2]
        roomlen_low = arguments[3] 
        roomlen_high = arguments[4]
        
        roomwid_mean = arguments[5]
        roomwid_low = arguments[6]
        roomwid_high = arguments[7]
        
        roomhei_mean = arguments[8]
        roomhei_low = arguments[9]
        roomhei_high = arguments[10]
        
        micx = arguments[11]
        micy = arguments[12]
        michei = arguments[13]
        
        # generate random room specifications, including absorption factor.
        alpha = random.uniform(alpha_low, alpha_high)

        room_length = roomlen_mean + np.random.uniform(roomlen_low, roomlen_high)
        room_width = roomwid_mean + np.random.uniform(roomwid_low, roomwid_high)
        room_height = roomhei_mean + np.random.uniform(roomhei_low, roomhei_high)

        microphone_x = np.random.uniform(micx, room_width - micx)
        microphone_y = np.random.uniform(micy, room_length - micy)
        microphone_height = 1.50 + np.random.uniform(-michei, michei)

        r = 0.5 * np.sqrt(np.random.uniform(0, 1))
        theta = np.random.uniform(0, 1) * 2 * np.pi
        source_x = microphone_x + r * np.cos(theta)
        source_y = microphone_y + r * np.sin(theta)
        source_height = 1.80 + np.random.uniform(-0.25, 0.25)
        
        fs = self.sampling_rate
        
        # create the room based on the specifications simulated above
        room = pra.ShoeBox([room_width, room_length, room_height],
                           fs=fs, #fs should be an integer, not array. 
                           max_order=17,
                           absorption=alpha)
        

        # add recording at source, and a random microphone to room
        room.add_source([source_x, source_y, source_height], signal=recording)
        R = np.array([[microphone_x], [microphone_y], [microphone_height]])
        room.add_microphone_array(pra.MicrophoneArray(R, room.fs))
        room.image_source_model(use_libroom=True)
        room.simulate()

        # return the reverberation convolved signal
        return room.mic_array.signals[0, :]


    def volume_perturb(self,recording, *args):#self added so sampling_rate is correct
        """
        Select a gain in decibels randomly and add to recording

        :param recording:
        :return: Augmented recording
        """
        arguments = self.augment_args['volume_perturb']
        rand_low = arguments[0]
        rand_high = arguments[1]
        
        gain = np.random.randint(low=rand_low, high=rand_high)
        recording *= 10. ** (gain / 20.)
        return recording


    def add_wn(self,recording, *args):#self added so sampling_rate is correct
        """
        Add wn white noise with random variance to recording

        :param recording:
        :return: Augmented recording
        """
        arguments = self.augment_args['add_wn']
        var_low = arguments[0]
        var_high = arguments[1]
        
        # Normalize recording before adding wn
        mean = np.mean(recording)
        std = np.std(recording)

        recording = (recording - mean) / std
        variance = np.random.uniform(low=var_low, high=var_high)
        noise = np.random.normal(0, random.uniform(0, variance), len(recording))

        # append noise to signal
        return recording + noise

    def tempo_perturb(self, recording, *args):

        # Factor for speeding up or slowing down the tempo
        #   if < 1 : slow down
        #   if > 1 : speed up
        
        arguments = self.augment_args['tempo_perturb']
        rand_low = arguments[0]
        rand_high = arguments[1]

        rand = random.uniform(rand_low,rand_high) # To give some randomness from an interval. Not sure if needed
        tempo_factor = 1.0 + rand

        return librosa.effects.time_stretch(recording, tempo_factor) # load a local WAV file

    @staticmethod
    def choose_augmentation_scheme(list_of_augmentations,augment_prob_dir):
        
        if augment_prob_dir == None:
            n_augments = random.randint(0, len(list_of_augmentations))
            augmentations_to_apply = random.sample(
                list_of_augmentations, n_augments)

            augmentation_scheme = []
            for augmentation in list_of_augmentations:
                if augmentation in augmentations_to_apply:
                    augmentation_scheme.append(augmentation)

        
        else:
            augmentation_scheme = []
            for aug in list_of_augmentations:
                if random.uniform(0,1) < augment_prob_dir[aug]: 
                    augmentation_scheme += [aug]
                
        return augmentation_scheme

    def spec_augment(self, spectrogram):
        
        arguments = self.augment_args['spec_augment']
        time_warping_para= arguments[0]
        frequency_masking_para= arguments[1]
        time_masking_para= arguments[2]
        frequency_mask_num= arguments[3]
        time_mask_num= arguments[4]
        
        v = spectrogram.shape[0]
        tau = spectrogram.shape[1]
        augmented_spectrogram = spectrogram

        try:
            for i in range(frequency_mask_num):
                f = np.random.uniform(low=0.0, high=frequency_masking_para)
                f = int(f)
                f0 = random.randint(0, v - f)
                augmented_spectrogram[f0:f0 + f, :] = 0

            for i in range(time_mask_num):
                t = np.random.uniform(low=0.0, high=time_masking_para)
                t = int(t)
                t0 = random.randint(0, tau - t)
                augmented_spectrogram[:, t0:t0 + t] = 0
        except ValueError:
            pass

        return augmented_spectrogram


class InferenceSpectrogramAudioParser(AudioParser):

    def __init__(self, audio_config=None, context=20):
        # inits all audio configs
        super(InferenceSpectrogramAudioParser, self).__init__(audio_config)

        self.n_fft = int(self.sampling_rate * self.window_size)
        self.hop_length = int(self.sampling_rate * self.window_stride)
        self.context = context
        self.dataset_mean = 5.492418704733003
        self.dataset_std = 1.7552755216970917
        self.input_mean = 0
        self.input_std = 0
        self.alpha = 0
        self.alpha_increment = 0.1  # Corresponds to stop relying on dataset stats after 4sec
        self.nr_recordings = 0
        self.nr_frames = context * 2 + 5

    def parse_audio(self, part_of_recording, is_last=False):

        # Ignore last and
        if is_last and len(part_of_recording) < 320:
            if is_last:
                self.reset()
            return []

        self.alpha += self.alpha_increment

        D = librosa.stft(part_of_recording, n_fft=self.n_fft, hop_length=self.hop_length,
                         win_length=self.n_fft, window=self.window, center=False)

        spect, phase = librosa.magphase(D)
        # S = log(S+1)
        spect = np.log1p(spect)

        self.input_mean = (self.input_mean + np.mean(spect)) / 2
        self.input_std = (self.input_std + np.std(spect)) / 2

        # Whenever alpha is done, rely only on input stats
        if self.alpha < 1.0:
            mean = self.input_mean * self.alpha + (1 - self.alpha) * self.dataset_mean
            std = self.input_std * self.alpha + (1 - self.alpha) * self.dataset_std
        else:
            mean = self.input_mean
            std = self.input_std

        spect -= mean
        spect /= std
        spect = torch.FloatTensor(spect)

        if is_last:
            self.reset()

        return spect

    def reset(self):
        self.input_mean = 0
        self.input_std = 0
        self.alpha = 0
        self.nr_recordings = 0
