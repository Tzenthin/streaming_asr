import logging
import os
import numpy as np
import yaml
import time
import copy
from collections import deque
from collections import defaultdict
from wenet.utils.file_utils import read_symbol_table
from wenet.transformer.asr_model_streaming import init_asr_model
from wenet.utils.checkpoint import load_checkpoint
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import datetime
from grpc_FeaturePipeline import Feature_Pipeline
from wenet.utils.common import (IGNORE_ID, add_sos_eos, log_add,
                                remove_duplicates_and_blank, th_accuracy,
                                reverse_pad_list)
from wenet.utils.mask import (make_pad_mask, mask_finished_preds,
                              mask_finished_scores, subsequent_mask)
from torch.nn.utils.rnn import pad_sequence
from ppasr.utils.text_utils import PunctuationExecutor


class ASR_Model():

    def __init__(self, model_config): #, Feat_Pipeline):
        #with open(model_config_path, 'r') as fin:
        #    self.configs = yaml.load(fin, Loader=yaml.FullLoader)
        self.configs = model_config
        symbol_table = read_symbol_table(self.configs['dict_path'])
        #os.environ['CUDA_VISIBLE_DEVICES'] = str(self.configs['gpu'])
        #decode_conf = copy.deepcopy(self.configs['data_conf'])
        #decode_conf['filter_conf']['max_length'] = 102400
        #decode_conf['filter_conf']['min_length'] = 0
        #decode_conf['filter_conf']['token_max_length'] = 102400
        #decode_conf['filter_conf']['token_min_length'] = 0
        #use_cuda = self.configs['gpu'] >= 0 and torch.cuda.is_available()
        #self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.device = torch.device('cpu')
        # convert num to symbles 
        self.num2sym_dict = {}
        with open(self.configs['dict_path'], 'r') as fin:
            for line in fin:
                arr = line.strip().split()
                assert len(arr) == 2
                self.num2sym_dict[int(arr[1])] = arr[0]
        self.eos = len(self.num2sym_dict) - 1
        self.sos = self.eos
        self.ignore_id = IGNORE_ID
        self.reverse_weight = self.configs['reverse_weight']
        self.ctc_weight = self.configs['model_conf']['ctc_weight']

        self.wenet_asr = init_asr_model(self.configs)
        load_checkpoint(self.wenet_asr, self.configs['model_path'])
        self.wenet_asr = self.wenet_asr.to(self.device)
        self.wenet_asr.eval()
        
        self.feat_pipeline = Feature_Pipeline(self.configs)
        self.exist_endpoint = False
        #self.num_frames = 0
        self.frames_stride = self.configs['decoding_chunk_size'] * self.wenet_asr.encoder.embed.subsampling_rate
        self.right_context = self.wenet_asr.encoder.embed.right_context + 1
        #self.feats_queue = torch.tensor()
        self.feats_queue = torch.zeros([1,1,80])
        self.offset = 0
        self.required_cache_size = self.configs['decoding_chunk_size'] * self.configs['num_decoding_left_chunks']
        self.subsampling_cache: Optional[torch.Tensor] = None
        self.elayers_output_cache: Optional[List[torch.Tensor]] = None
        self.conformer_cnn_cache: Optional[List[torch.Tensor]] = None
        self.encoder_out = torch.tensor(0.)
        self.result = ''
        #self.hyps = []
        self.cur_hyps = [(tuple(), (0.0, -float('inf')))]
        #self.ctc_probs_ = torch.tensor(0.)
        self.beam_size = self.configs['beam_size']
        #self.device = device = torch.device('cuda' if use_cuda else 'cpu')
        self.device = torch.device('cpu')
        self.decoding_once = False
        
        torch.set_num_threads(8)
        torch.set_num_interop_threads(1)

        self.pun_executor = PunctuationExecutor(model_dir='AddPunctuation/pun_models')


    def set_input_pipeline(self, waveform_byte):  #num_mel_bins=80, frame_length=25, frame_shift=10, dither=0.0, sample_rate=16000):
        self.exist_endpoint = self.feat_pipeline.AcceptWaveform(waveform_byte)
    
    def get_wave_len_in_feat_pipeline(self): 
        return self.feat_pipeline.get_waveform_len()

    def get_wav_before_endpoint(self):
        assert self.endpoint_detected() == True
        return self.feat_pipeline.GetFirstWav()

    def init_decoding(self):
        self.exist_endpoint = False
        #self.num_frames = 0
        self.feats_queue = torch.zeros([1,1,80])
        self.offset = 0
        self.subsampling_cache: Optional[torch.Tensor] = None
        self.elayers_output_cache: Optional[List[torch.Tensor]] = None
        self.conformer_cnn_cache: Optional[List[torch.Tensor]] = None
        self.encoder_out = torch.tensor(0.)
        self.hyps = []
        self.feat_pipeline.Reset()
        self.result = ''
        self.cur_hyps = [(tuple(), (0.0, -float('inf')))]

    '''
    def _ctc_prifix_beam_search(ctc_probs):
        #ctc_probs = ctc_probs.squeeze(0)
        #self.ctc_probs_append(ctc_probs)
        # cur_hyps: (prefix, (blank_ending_score, none_blank_ending_score))
        cur_hyps = [(tuple(), (0.0, -float('inf')))] 
        # 2. CTC beam search step by step 
        for t in range(0, maxlen):
            logp = ctc_probs[t]  # (vocab_size,)
            # key: prefix, value (pb, pnb), default value(-inf, -inf)
            next_hyps = defaultdict(lambda: (-float('inf'), -float('inf')))
            # 2.1 First beam prune: select topk best
            top_k_logp, top_k_index = logp.topk(self.beam_size)  # (beam_size,)
            for s in top_k_index:
                s = s.item()
                ps = logp[s].item()
                for prefix, (pb, pnb) in cur_hyps:
                    last = prefix[-1] if len(prefix) > 0 else None
                    if s == 0:  # blank
                        n_pb, n_pnb = next_hyps[prefix]
                        n_pb = log_add([n_pb, pb + ps, pnb + ps])
                        next_hyps[prefix] = (n_pb, n_pnb)
                    elif s == last:
                        #  Update *ss -> *s;
                        n_pb, n_pnb = next_hyps[prefix]
                        n_pnb = log_add([n_pnb, pnb + ps])
                        next_hyps[prefix] = (n_pb, n_pnb)
                        # Update *s-s -> *ss, - is for blank
                        n_prefix = prefix + (s, )
                        n_pb, n_pnb = next_hyps[n_prefix]
                        n_pnb = log_add([n_pnb, pb + ps])
                        next_hyps[n_prefix] = (n_pb, n_pnb)
                    else:
                        n_prefix = prefix + (s, )
                        n_pb, n_pnb = next_hyps[n_prefix]
                        n_pnb = log_add([n_pnb, pb + ps, pnb + ps])
                        next_hyps[n_prefix] = (n_pb, n_pnb)
            # 2.2 Second beam prune
            next_hyps = sorted(next_hyps.items(), 
                               key=lambda x: log_add(list(x[1])),
                               reverse=True)
            cur_hyps = next_hyps[:self.beam_size]
        hyps = [(y[0], log_add([y[1][0], y[1][1]])) for y in cur_hyps]
        return hyps #, encoder_out
    '''

    def _chunk_ctc_prefix_beam_search(self, encoder_out): #, cur_hyps): # , encoder_mask):
        #assert speech.shape[0] == speech_lengths.shape[0]
        #assert decoding_chunk_size != 0
        #batch_size = speech.shape[0]
        #encoder
        maxlen = encoder_out.size(1)
        ctc_probs = self.wenet_asr.ctc.log_softmax(encoder_out) #(Batchsize==1, maxlen, vocab_size)
        ctc_probs = ctc_probs.squeeze(0)
        
        #if not self.ctc_probs.equal(torch.tensor(0.)):
        #    self.ctc_probs = torch.cat((self.ctc_probs, ctc_probs), 0)
        #else: self.ctc_probs = ctc_probs
        
        # cur_hyps: (prefix, (blank_ending_score, none_blank_ending_score))
        #cur_hyps = [(tuple(), (0.0, -float('inf')))] 
        # 2. CTC beam search step by step 
        for t in range(0, maxlen):
            logp = ctc_probs[t]  # (vocab_size,)
            # key: prefix, value (pb, pnb), default value(-inf, -inf)
            next_hyps = defaultdict(lambda: (-float('inf'), -float('inf')))
            # 2.1 First beam prune: select topk best
            top_k_logp, top_k_index = logp.topk(self.beam_size)  # (beam_size,)
            for s in top_k_index:
                s = s.item()
                ps = logp[s].item()
                for prefix, (pb, pnb) in self.cur_hyps:
                    last = prefix[-1] if len(prefix) > 0 else None
                    if s == 0:  # blank
                        n_pb, n_pnb = next_hyps[prefix]
                        n_pb = log_add([n_pb, pb + ps, pnb + ps])
                        next_hyps[prefix] = (n_pb, n_pnb)
                    elif s == last:
                        #  Update *ss -> *s;
                        n_pb, n_pnb = next_hyps[prefix]
                        n_pnb = log_add([n_pnb, pnb + ps])
                        next_hyps[prefix] = (n_pb, n_pnb)
                        # Update *s-s -> *ss, - is for blank
                        n_prefix = prefix + (s, )
                        n_pb, n_pnb = next_hyps[n_prefix]
                        n_pnb = log_add([n_pnb, pb + ps])
                        next_hyps[n_prefix] = (n_pb, n_pnb)
                    else:
                        n_prefix = prefix + (s, )
                        n_pb, n_pnb = next_hyps[n_prefix]
                        n_pnb = log_add([n_pnb, pb + ps, pnb + ps])
                        next_hyps[n_prefix] = (n_pb, n_pnb)
            # 2.2 Second beam prune
            next_hyps = sorted(next_hyps.items(), 
                               key=lambda x: log_add(list(x[1])),
                               reverse=True)
            self.cur_hyps = next_hyps[:self.beam_size]
        hyps = [(y[0], log_add([y[1][0], y[1][1]])) for y in self.cur_hyps]
        return hyps #, encoder_out


    def advance_decoding(self):
        
        #current_wav_len = self.feat_pipeline.get_waveform_len()
        #if current_wav_len < self.configs['engine_sample_rate_hertz'] * 0.05:
        #    return 

        feat_read = self.feat_pipeline.ReadFeats() #frames_stride) #ight_context)
        if feat_read==None: 
            return
        #print('待计算的特征长度：', feat_read.shape)
        if self.feats_queue.shape[1] == 1:
            self.feats_queue = feat_read
        else:
            self.feats_queue = torch.cat((self.feats_queue, feat_read), 1) #[1, 67, 80]

        encoder_frames = self.frames_stride #+self.right_context
        if self.exist_endpoint == True:
            feat_to_encoder = self.feats_queue
            self.feats_queue = torch.zeros([1,1,80])
        else:
            if self.feats_queue.size(1) >= encoder_frames:
                #if self.exist_endpoint == True:
                #    feat_to_encoder = self.feats_queue
                #    self.feats_queue = torch.zeros([1,1,80])
                feat_to_encoder = self.feats_queue[:, 0:encoder_frames, :]
                self.feats_queue = self.feats_queue[:, self.frames_stride:, :]
            else:  
                # 如果不是端点，且收集到的特征长度还没有达到设定的chunksize大小，则不做处理，等待下一个流的拼接
                # 如果音频最后没有端点，且最后的一段有声段很短，直接return啥都不做会出bug
                # 如果25ms 10ms，则20帧约200ms, chunksize=16，约降采样为4，则67帧，越64*10=640ms解码一次。
                # chunksize=8 ,320ms解码一次 
                return 

        if feat_to_encoder.size(1) > 16: # 设置的最小解码帧数, 因为卷积核的大小为3*3
            (y, self.subsampling_cache, self.elayers_output_cache,
              self.conformer_cnn_cache) = self.wenet_asr.encoder.forward_chunk(feat_to_encoder, self.offset,
                                                   self.required_cache_size,
                                                   self.subsampling_cache,
                                                   self.elayers_output_cache,
                                                   self.conformer_cnn_cache)
            self.offset += y.size(1)

            if self.encoder_out.equal(torch.tensor(0.)):
                self.encoder_out = y
            else: self.encoder_out = torch.cat((self.encoder_out, y), 1)
            hyps = self._chunk_ctc_prefix_beam_search(y) #, mask)
            hyps = list(hyps[0][0]) #.tolist()
            result = self._num2sym(hyps)
            self.result = 'partial'+'+++'+result
            print('advance decoding: ', self.result)
            self.decoding_once = True
             
        else: 
            #print('解码的feature过短, 代码bug！')
            #assert 0==1
            #result = ''
            return

    def finalize_rescoring(self):
        #current_wav_len = self.feat_pipeline.get_waveform_len()
        #if current_wav_len < self.configs['engine_sample_rate_hertz'] * 0.2:
        #    return 
        if len(self.result.split('+++')[-1])<1: #如果ctc解码没有结果的话，就略过rescoring
            return
        #print(self.encoder_out.shape)
        #assert 0==1
        encoder_out = self.encoder_out #torch.cat(self.encoder_out, 1)
        hyps = [(y[0], log_add([y[1][0], y[1][1]])) for y in self.cur_hyps]
        #hyps = self.hyps #torch.cat(self.hyps, 1)
        masks = torch.ones(1, encoder_out.size(1), device=encoder_out.device, dtype=torch.bool)
        masks = masks.unsqueeze(1)
        assert len(hyps) == self.beam_size
        hyps_pad = pad_sequence([
            torch.tensor(hyp[0], device=self.device, dtype=torch.long)
            for hyp in hyps
            ], True, self.ignore_id)  # (beam_size, max_hyps_len)
        ori_hyps_pad = hyps_pad
        hyps_lens = torch.tensor([len(hyp[0]) for hyp in hyps],
                                 device=self.device,
                                 dtype=torch.long)  # (beam_size,)
        hyps_pad, _ = add_sos_eos(hyps_pad, self.sos, self.eos, self.ignore_id)
        hyps_lens = hyps_lens + 1  # Add <sos> at begining
        encoder_out = encoder_out.repeat(self.beam_size, 1, 1)
        encoder_mask = torch.ones(self.beam_size,
                                  1,
                                  encoder_out.size(1),
                                  dtype=torch.bool,
                                  device=self.device)
        # used for right to left decoder
        r_hyps_pad = reverse_pad_list(ori_hyps_pad, hyps_lens, self.ignore_id)
        r_hyps_pad, _ = add_sos_eos(r_hyps_pad, self.sos, self.eos,
                                    self.ignore_id)
        with torch.no_grad():
            decoder_out, r_decoder_out, _ = self.wenet_asr.decoder(
                    encoder_out, encoder_mask, hyps_pad, hyps_lens, r_hyps_pad,
                    self.reverse_weight)  # (beam_size, max_hyps_len, vocab_size)
        decoder_out = torch.nn.functional.log_softmax(decoder_out, dim=-1)
        decoder_out = decoder_out.cpu().numpy()
        # r_decoder_out will be 0.0, if reverse_weight is 0.0 or decoder is a
        # conventional transformer decoder.
        r_decoder_out = torch.nn.functional.log_softmax(r_decoder_out, dim=-1)
        r_decoder_out = r_decoder_out.cpu().numpy()
        # Only use decoder score for rescoring
        best_score = -float('inf')
        best_index = 0
        for i, hyp in enumerate(hyps):
            score = 0.0
            for j, w in enumerate(hyp[0]):
                score += decoder_out[i][j][w]
            score += decoder_out[i][len(hyp[0])][self.eos]
            # add right to left decoder score
            if self.reverse_weight > 0:
                r_score = 0.0
                for j, w in enumerate(hyp[0]):
                    r_score += r_decoder_out[i][len(hyp[0]) - j - 1][w]
                r_score += r_decoder_out[i][len(hyp[0])][self.eos]
                score = score * (1 - self.reverse_weight) + r_score * self.reverse_weight
            # add ctc score 
            score += hyp[1] * self.ctc_weight
            if score > best_score:
                best_score = score
                best_index = i
        #return hyps[best_index][0], best_score
        #print('resoring :', hyps)
        #hyps_tmp = [hyp.tolist() for hyp in  hyps[best_index][0][0]]
        hyps_tmp = hyps[best_index][0]
        #print('resoring :', hyps_tmp)
        #assert 0==1
        result = self._num2sym(hyps_tmp)
        #self.result = 'final+++'+result
        #pr_result = self.pr(result)
        self.result = 'final'+'+++'+result
        self.decoding_once = True
        #return result
        print('final rescoring: ', self.result)

    def pr(self, text):
        #pun_executor = PunctuationExecutor(model_dir='pun_models')
        result = self.pun_executor(text)
        return result


    def ctc_rescoring_decoding(self, audio):
        waveform = np.frombuffer(audio, dtype=np.int16)
        wav_duration = len(waveform)/self.configs['engine_sample_rate_hertz']
        waveform = torch.from_numpy(waveform).float().unsqueeze(0)
        #waveform = waveform.to(self.device)
        waveform_feat, feat_length = self.feat_pipeline._extract_feature(waveform)
        with torch.no_grad():
            hyps, scores = self.wenet_asr.attention_rescoring(
                           waveform_feat,
                           feat_length,
                           beam_size=self.configs['beam_size'],
                           decoding_chunk_size=-1,
                           num_decoding_left_chunks=self.configs['num_decoding_left_chunks'],
                           ctc_weight=self.configs['model_conf']['ctc_weight'],
                           simulate_streaming=False,
                           reverse_weight=0.0
                           )   # 对于双向的decoder才有reverse的权重
            #hyps = [hyp.tolist() for hyp in hyps[0]]
            #print(hyps)
            result = self._num2sym(hyps)
            #print(result)
            result = self.pr(result)
        return result+'+++'+str(wav_duration)+'s'

    def attention_decoding(self, audio):
        waveform = np.frombuffer(audio, dtype=np.int16)
        wav_duration = len(waveform)/self.configs['engine_sample_rate_hertz']
        waveform = torch.from_numpy(waveform).float().unsqueeze(0)
        #waveform = waveform.to(self.device)
        waveform_feat, feat_length = self.feat_pipeline._extract_feature(waveform)
        #model = self._get_model()
        ts = time.time()
        with torch.no_grad():
            hyps, scores = self.wenet_asr.recognize(
                                 waveform_feat,
                                 feat_length,
                                 beam_size=self.configs['beam_size'],
                                 decoding_chunk_size=-1,
                                 num_decoding_left_chunks=-1,
                                 simulate_streaming=False
                                 )  #args.simulate_streaming)
            #print(hyps)
            hyps = [hyp.tolist() for hyp in hyps[0]]
            #print(hyps)
            result = self._num2sym(hyps)
            result = self.pr(result)
        #print(result) #, scores)
        te = time.time()
        #print('解码时间：',te-ts)
        #return result
        return result+'+++'+str(wav_duration)+'s'

    def ctc_greedy_search(self, audio):
        waveform = np.frombuffer(audio, dtype=np.int16)
        wav_duration = len(waveform)/self.configs['engine_sample_rate_hertz']
        waveform = torch.from_numpy(waveform).float().unsqueeze(0)
        waveform_feat, feat_length = self.feat_pipeline._extract_feature(waveform)
        with torch.no_grad():
            hyps, scores = self.wenet_asr.ctc_greedy_search(
                                 waveform_feat,
                                 feat_length,
                                 decoding_chunk_size=-1,
                                 num_decoding_left_chunks=-1,
                                 simulate_streaming=False
                                 )
            #hyps = [hyp.tolist() for hyp in hyps[0]]
            #print(hyps)
            result = self._num2sym(hyps[0])
            result = self.pr(result)
        #return result
        return result+'+++'+str(wav_duration)+'s'

    def endpoint_detected(self):
        return self.exist_endpoint
    
    def get_output(self):
        return self.result

    def _num2sym(self, hyps):
        content = ''
        for w in hyps:
            if w == self.eos:
                break
            content += self.num2sym_dict[w]
        return content




