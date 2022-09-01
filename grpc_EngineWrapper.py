# Wrapper class that adds pre/post-processing to STT engine

import logging
from attrdict import AttrDict
import os
from datetime import datetime
import wave
from text2digits import text2digits
import configargparse

class EngineWrapper(object):
    def __init__(self, config, engine):
        """ Initializes the object.

        Args:
            config (dict): Key, value pair of configuration values.

        Returns:
            STTEngine object with pre-processing decorators.
        """
        parser = configargparse.ArgumentParser(description="STT GRPC engine wrapper.",
                default_config_files=["config"])
        parser.add_argument('--savewav', default='',
                help="Save .wav files of utterences to given directory.")
        ARGS, _ = parser.parse_known_args()
        args = vars(ARGS)

        self.config = AttrDict({**args, **config})
        if self.config.savewav: os.makedirs(self.config.savewav, exist_ok=True)
        self.engine = engine
        self.logger = logging.getLogger('wrapper.save_post')

    def post_fun(self, result):
        """Performs text2digit conversion on transcript portion of result"""
        if isinstance(result, dict):
            text = result.get('transcript', '')
            result['transcript'] = text2digits.Text2Digits().convert(text)
            return(result)
        else:
            return(text2digits.Text2Digits().convert(result))

    def decode_audio(self, audio, method):
        """ Decodes an audio segment

        Args:
            audio ([bytes]): Audio data to be transcribed. 

        Returns:
            Decoded string.
        """
        if self.config.savewav:
            self.save_wave(audio)
        text = self.engine.decode_audio(audio, method)
        return self.post_fun(text)


    def get_stream(self, result_queue):
        """ Establishes stream to model.

        Args:
            result_queue:   Queue to pass completed results to.

        Returns:
            A stream object to refer back.
        """
        # FIXME: Should include some guard agains very long audio!
        if self.config.savewav:
            self.audio = bytearray()

        # FIXME: Poor workaround for post_fun?
        pre_wrapper = result_queue.put
        def put_wrapper(item, *args, **kwargs):
            pre_wrapper(self.post_fun(item), *args, **kwargs)
        result_queue.put = put_wrapper
        
        return self.engine.get_stream(result_queue)
        #return {'stream': self.engine.get_stream(result_queue), 'result_queue':result_queue}


    def feed_audio_data(self, stream, audio):
        """ Adds audio data to given stream

        Args:
            stream: Object returned by setup_stream.
            audio ([bytes]): Audio data to be transcribed. 
        """
        # FIXME: Will different stream have an issue?
        if self.config.savewav:
            self.audio.extend(audio)
        return self.engine.feed_audio_data(stream, audio)

    def finish_decoding(self, stream):
        if self.config.savewav:
            self.audio.extend(audio)
        return self.engine.finish_decoding(stream)


    def finish_stream(self, stream):
        """ Finishes decoding destroying stream.

        Args:
            stream: Object returned by setup_stream.

        """
        if self.config.savewav:
            self.save_wave(self.audio)
        return self.engine.finish_stream(stream)

    def save_wave(self, audio):
        fn = os.path.join(self.config.savewav, datetime.now().strftime("savewav_%Y-%m-%d_%H-%M-%S_%f.wav"))
        with wave.open(fn, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.config.sample_rate_hertz)
            wf.writeframes(audio)
            self.logger.info('Audio of length {} saved.'.format(len(audio)))

    def __getattr__(self, attr):
        """ Passess all non-implemented method to engin
        """
        return getattr(self.engine, attr)
