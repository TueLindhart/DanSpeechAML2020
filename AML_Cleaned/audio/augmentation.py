from abc import ABC, abstractmethod

import numpy as np
import random

import librosa
import pyroomacoustics as pra


class DataAugmenter(ABC):
    """
    Abstract class for data augmentations
    """

    @abstractmethod
    def augment(self, recording):
        pass


class DanSpeechAugmenter(DataAugmenter):
    """
    Class that implements the DanSpeech Augmentation scheme
    """

    def __init__(self, sampling_rate, augmentations_list=None, argument_dict=None):
        self.sampling_rate = sampling_rate 

        # Allow user to specify a list of augmentations
        # otherwise default to danspeech schema.
        if not augmentations_list:
            self.augmentations_list = [
                "speed_perturb",
                "room_reverb",
                "volume_perturb",
                "add_wn",
                "shift_perturb",#Before was not strings but object.  
                "tempo_perturb"
            ]
        #if not argument_dict:
            

    def augment(self, recording):
        scheme = self.choose_augmentation_scheme()#augmentions_list is removed inside the function 

        if len(scheme) > 0:
            for augmentation in scheme:
                augmentor = getattr(self, augmentation)
                recording = augmentor(recording)

        return recording

    def choose_augmentation_scheme(self):
        """
        Chooses a valid danspeech augmentation based on the ordered
        list of augmentations

        :param list_of_augmentations: Ordered list of augmentation functions
        :return: A valid danspeech augmentation scheme
        """
        n_augments = random.randint(0, len(self.augmentations_list))
        augmentations_to_apply = random.sample(
            self.augmentations_list, n_augments)

        augmentation_scheme = []
        for augmentation in self.augmentations_list:
            if augmentation in augmentations_to_apply:
                augmentation_scheme.append(augmentation)

        return augmentation_scheme

    #The following functions are indented to be are part of the class DanSpeechAugmentor class
    
    def speed_perturb(self,recording, rand_low = 0.9,rand_high = 1.1, *args): #self added so sampling_rate is correct
        """
        Select up/down-sampling randomly between 90% and 110% of original sample rate

        :param recording: Recording to be augmented
        :param sampling_rate: Sampling rate of the recording
        :return: Augmented recording
        """
        new_sample_rate = self.sampling_rate * random.choice([rand_low,rand_high]) # again the sampling rate should simply be a number
        return librosa.core.resample(recording, self.sampling_rate, new_sample_rate)


    def shift_perturb(self,recording, rand_low = -50, rand_high = 50, *args):#self added so sampling_rate is correct
        """
        Shifts the audio recording randomly in time.

        :param recording: Recording to be augmented
        :param sampling_rate: Sampling rate of the recording
        :return: Augmented recording
        """

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


    def room_reverb(self,recording, alpha_low = 0, alpha_high = 0.4, roomlen_low = 2, roomlen_high = 12, roomwid_low = 2,
                    roomwid_high = 6, roomhei_low = -0.5, roomhei_high = 0.5, micx = 0.5, micy = 0.5, michei = 0.5, *args):
                    #self added so sampling_rate is correct
        """
        Perturb signal with room reverberations in a randomly generated shoebox room.

        :param recording: Recording to be augmented
        :param sampling_rate: Sampling rate of the recording
        :return: Augmented recording
        """

        # generate random room specifications, including absorption factor.
        alpha = random.uniform(alpha_low, alpha_high)

        room_length = np.random.uniform(roomlen_low, roomlen_high)
        room_width = np.random.uniform(roomwid_low, roomwid_high)
        room_height = 3.0 + np.random.uniform(roomhei_low, roomhei_high)

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


    def volume_perturb(self,recording, rand_low=5,rand_high=30, *args):#self added so sampling_rate is correct
        """
        Select a gain in decibels randomly and add to recording

        :param recording:
        :return: Augmented recording
        """
        gain = np.random.randint(low=rand_low, high=rand_high)
        recording *= 10. ** (gain / 20.)
        return recording


    def add_wn(self,recording, var_low = 0.5, var_high = 1.8, *args):#self added so sampling_rate is correct
        """
        Add wn white noise with random variance to recording

        :param recording:
        :return: Augmented recording
        """
        # Normalize recording before adding wn
        mean = np.mean(recording)
        std = np.std(recording)

        recording = (recording - mean) / std



        variance = np.random.uniform(low=var_low, high=var_high)
        noise = np.random.normal(0, random.uniform(0, variance), len(recording))

        # append noise to signal
        return recording + noise

    def tempo_perturb(self, recording, rand_low = -0.25, rand_high = 0.25, *args):

        # Factor for speeding up or slowing down the tempo
        #   if < 1 : slow down
        #   if > 1 : speed up

        rand = random.uniform(rand_low,rand_high) # To give some randomness from an interval. Not sure if needed
        tempo_factor = 1.0 + rand

        return librosa.effects.time_stretch(recording, tempo_factor) # load a local WAV file
