# Wrapper class that adds vad pre-processing to STT engine

import logging
from attrdict import AttrDict
import webrtcvad
import collections

class VADAudio():
    """Represents VAD segmented Audio"""
    SAMPLE_WIDTH = 2 # Number of bytes for each sample
    CHANNELS = 1

    #def __init__(self, aggressiveness, rate, frame_duration_ms=30, padding_ms=200, padding_ratio=0.4):
    def __init__(self, aggressiveness, rate, frame_duration_ms=30, padding_ms=400, padding_ratio=0.75):
        """Initializes VAD with given aggressivenes and sets up internal queues"""
        self.vad = webrtcvad.Vad(aggressiveness)
        self.rate = rate
        self.frame_duration_ms = frame_duration_ms
        self._frame_length = int( rate * (frame_duration_ms/1000.0) * self.SAMPLE_WIDTH )
        self._buffer_queue = collections.deque()
        self.ring_buffer = collections.deque(maxlen = padding_ms // frame_duration_ms)
        self._ratio = padding_ratio
        self.triggered = False

    def add_audio(self, audio):
        """Adds new audio to internal queue"""
        for x in audio:
            self._buffer_queue.append(x)

    def frame_generator(self):
        """Generator that yields audio frames of frame_duration_ms"""
        while len(self._buffer_queue) > self._frame_length:
            frame = bytearray()
            for _ in range(self._frame_length):
                frame.append(self._buffer_queue.popleft())
            yield bytes(frame)

    def vad_collector(self):
        """Generator that yields series of consecutive audio frames comprising each utterence, separated by yielding a single None.
            Determines voice activity by ratio of frames in padding_ms. Uses a buffer to include padding_ms prior to being triggered.
            Example: (frame, ..., frame, None, frame, ..., frame, None, ...)
                      |---utterence---|        |---utterence---|
        """
        for frame in self.frame_generator():
            is_speech = self.vad.is_speech(frame, self.rate)
            if not self.triggered:
                self.ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in self.ring_buffer if speech])
                if num_voiced > self._ratio * self.ring_buffer.maxlen:
                    self.triggered = True
                    for f, s in self.ring_buffer:
                        yield f
                    self.ring_buffer.clear()
            else:
                yield frame
                self.ring_buffer.append((frame, is_speech))
                num_unvoiced = len([f for f, speech in self.ring_buffer if not speech])
                if num_unvoiced > self._ratio * self.ring_buffer.maxlen:
                    self.triggered = False
                    yield None
                    self.ring_buffer.clear()

    def endpoint_detect(self, audio):
        IsEndpoint = False
        audio_data = b''
        remained_audio = b''
        self.add_audio(audio)
        temp_audio = b''
        for frame in self.vad_collector():
            if frame is None:
                IsEndpoint = True
                if len(temp_audio)>0 and audio_data==b'':
                    audio_data = temp_audio
                    temp_audio = b''
            else:
                temp_audio += frame
        if IsEndpoint == True: remained_audio = temp_audio
        else: audio_data = temp_audio
        if audio_data == b'':
            IsEndpoint = False
            audio_data = remained_audio
            remained_audio = b''
        return audio_data, remained_audio, IsEndpoint




