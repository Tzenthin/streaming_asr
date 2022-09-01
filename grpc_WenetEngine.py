import logging
import argparse
import os
import numpy as np
import tempfile
import csv
import math
import yaml
import time
import copy
from grpc_STTEngine import STTEngine
from collections import deque
#from wenet.utils.file_utils import read_symbol_table
#from wenet.transformer.asr_model_streaming import init_asr_model
#from wenet.utils.checkpoint import load_checkpoint
#import torch
#import torchaudio
#import torchaudio.compliance.kaldi as kaldi
import datetime
import wave
from grpc_FeaturePipeline import Feature_Pipeline
from grpc_CoreModel import ASR_Model

class WenetEngine(STTEngine):
    #torch.set_num_threads(1)
    #torch.set_num_interop_threads(1)

    def __init__(self, model_config_path):
        self.logger = logging.getLogger('engine.wenet!')
        with open(model_config_path, 'r') as fin:
            self.configs = yaml.load(fin, Loader=yaml.FullLoader)
        logging.basicConfig(level=logging.DEBUG,format='%(asctime)s %(levelname)s %(message)s')
        #feat_pipeline = Feature_Pipeline(self.configs)
        self.model = ASR_Model(self.configs) #, feat_pipeline)

        self.models = deque(maxlen=self.configs['engine_max_decoders'])
        for i in range(self.configs['engine_max_decoders']):
            #asr_copy = copy.deepcopy(asr)
            self.models.append(self.model)
            self.logger.info('Model {} loaded.'.format(id(self.model)))
        self.streams = []
        self.DECODE_CHUNK_SIZE = self.configs['engine_sample_rate_hertz'] * 0.01

    def _get_model(self):
        if len(self.models):
            model = self.models.pop()
            self.logger.info('Model {} engaged.'.format(id(model)))
            return model
        else:
            for ix, s in enumerate(self.streams):
                if (time.time() - s['last_activity']) > self.configs['engine_max_inactivity_secs']:
                    model = s['model']
                    self.streams.pop(ix)
                    self.logger.info('Model {} force freed.'.format(id(model)))
                    return model
        #raise MemoryError
        return self.model

    def _free_model(self, model):
        self.models.append(model)
        self.logger.info('Model {} freed.'.format(id(model)))

    def _text_process(self, text):
        text = text.replace("[noise]", "")
        if self.phone_dict:
            ph_seq = ' '.join(self.phone_dict.get(w, w) for w in text.split())
        if self.sentence_piece:
            text = text.replace(' ', '').replace('_', ' ')
        if self.phone_dict:
            return({'transcript':text, 'phoneme': ph_seq})
        return text
    
    def decode_audio(self, audio, method):  # waveform是由torchaudio读取出来，并左移15位得到的数据
        if len(audio)<self.configs['engine_sample_rate_hertz']<0.1: #小于0.1秒，不解码
            return ''
        model = self._get_model()

        if method == 'ctc_greedy_search':
            result = model.ctc_greedy_search(audio)
        elif method == 'ctc_attention_rescoring':
            result = model.ctc_rescoring_decoding(audio)
        elif method == 'attention':
            result = model.attention_decoding(audio)
        else: 
            result = model.ctc_greedy_search(audio)
        if self.configs['save_wave'] and len(result)>0:
            self.save_wave_with_tran(audio, result)
        self._free_model(model)
        return result           


    def get_stream(self, result_queue):
        asr = self._get_model()
        asr.init_decoding()
        stream = {'model':asr,
                'current_audio':bytes(), 
                'chunk_size':0, 
                'total_audio_len':0,
                'last_activity':time.time(), 
                'intermediate':'', 
                'result_queue':result_queue}
        self.streams.append(stream)
        self.logger.info('Stream established to Model {}.'.format(id(asr)))
        return stream

    def feed_audio_data(self, stream, audio):
        asr = stream['model']
        stream['last_activity'] = time.time()
        #if stream['current_audio'] != None:
        #    stream['current_audio'] += audio
        #else: stream['current_audio'] = audio
        asr.set_input_pipeline(audio) # accept waveform(detect endpoint)
        #if asr.get_wave_len_in_feat_pipeline() > self.DECODE_CHUNK_SIZE:
        asr.advance_decoding()
        
        result = asr.get_output()
        if asr.decoding_once:  #self.result是在否解码获取的，避免重复获取解码文本，重复向客户端发送
            if len(result.split('+++')[-1])>0:
                stream['result_queue'].put(result)
            asr.decoding_once = False
        
        if asr.endpoint_detected():
            stream['current_audio'] += asr.get_wav_before_endpoint()
            if self.configs['save_wave'] and len(result.split('+++')[-1].strip())>3:
                self.save_wave_with_tran(stream['current_audio'], result.split('+++')[-1].strip())
            stream['current_audio'] = bytes()
            print('检测到端点')
            if asr.decoding_once:  #self.result是在否解码获取的，避免重复获取解码文本，重复向客户端发送
                if len(result.split('+++')[-1])>0:
                    stream['result_queue'].put(result)
                asr.decoding_once = False
            #if result.split
            asr.finalize_rescoring()
            result = asr.get_output()
            if asr.decoding_once:  #self.result是在否解码获取的，避免重复获取解码文本，重复向客户端发送
                if len(result.split('+++')[-1])>0:
                    stream['result_queue'].put(result)
                asr.decoding_once = False
        else:
            stream['current_audio'] += audio
            stream['last_activity'] = time.time()
        
        
        if asr.endpoint_detected():
            asr.init_decoding()

    def finish_decoding(self, stream):
        asr = stream['model']
        asr.advance_decoding()
        result = asr.get_output()
        stream['result_queue'].put(result)
        print('未检测到端点进行rescoring') 
        if self.configs['save_wave'] and len(result.split('+++')[-1].strip())>3:
            self.save_wave_with_tran(stream['current_audio'], result.split('+++')[-1].strip())
        
        asr.finalize_rescoring()
        result = asr.get_output()
        stream['result_queue'].put(result)
        
        asr.init_decoding()
        self._free_model(stream['model'])

    def get_partial(self, stream):
        asr = stream['model']
        num_frames_decoded = asr.decoder.num_frames_decoded()
        if num_frames_decoded > stream['prev_num_frames_decoded']:
            stream['prev_num_frames_decoded'] = num_frames_decoded
            result = self._text_process(asr.get_partial_output()["text"])
            stream['intermediate'] = result
        return stream['intermediate']

    def finish_stream(self, stream):
        """ Finishes decoding destroying stream.
        Args:
            stream: Object returned by setup_stream.
        """
        asr = stream['model']
        #asr.finalize_decoding()
        #result = ''
        #if asr.decoder.num_frames_decoded() > 0:
        #    result = self._text_process(asr.get_output()["text"])
        #stream['result_queue'].put(result)
        #stream['feat_pipeline'].get_adaptation_state(adaptation_state)
        self.logger.info('Audio of length {} processed in stream to Model {}.'.format(stream['total_audio_len'], id(asr)))
        self._free_model(stream['model'])

    def check_compatibility(self, config):
        """ Checks if engine is compatible with given config.
        Args:
            config: Key, value pairs of requested features.
        Returns:
            boolean, True if engine matches config.
        """
        if 'sample_rate_hertz' in config:
            return config['sample_rate_hertz'] == self.configs['engine_sample_rate_hertz']
        return True

    def save_wave(self, audio):
        if len(audio) >  self.configs['engine_sample_rate_hertz'] * 0.4:
            fn = os.path.join(self.configs['audio_save_path'], datetime.datetime.now().strftime("savewav_%Y-%m-%d_%H-%M-%S_%f.wav"))
            with wave.open(fn, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.configs['engine_sample_rate_hertz'])
                wf.writeframes(audio)
                self.logger.info('Audio of length {} saved.'.format(len(audio)))


    def save_wave_with_tran(self, audio, transcript):
        now_time = datetime.datetime.now() 
        year = str(now_time.year) #.split()[0].split('-')[0]
        month = str(now_time.month) #.split()[0].split('-')[1]
        day = str(now_time.day) #split()[0].split('-')[2]
        audio_dir = os.path.join(self.configs['audio_save_path'], year, month, day)
        wav_file = audio_dir+'/'+ datetime.datetime.now().strftime("%H-%M-%S-%f_") + transcript[0:20] + '_.wav'
        if not os.path.exists(audio_dir):
            os.makedirs(audio_dir)
            #datetime.now().strftime("savewav_%Y-%m-%d_%H-%M-%S_%f.wav"))
        with wave.open(wav_file, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.configs['engine_sample_rate_hertz'])
            wf.writeframes(audio)
            #self.logger.info('Audio of length {} saved.'.format(len(audio)))

