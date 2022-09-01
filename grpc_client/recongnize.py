import stt_pb2_grpc
import stt_pb2
import grpc
import time
import torchaudio
import argparse
import torch
import numpy 

def read_pcm(audio):
    with open(audio,'rb') as f:
        data = f.read()
    return data

if __name__ == '__main__':
    audio = 'test_audio/'+'BAC009S0764W0136.wav'
    #audio = 'test_audio/exp4/'+'D4-spk1-0.wav'
    print("Starting...")
    start_time = time.perf_counter()
    channel = grpc.insecure_channel("0.0.0.0:5511")
    stub = stt_pb2_grpc.STTStub(channel)
    #decode_mode = 'attention'
    #decode_mode = 'ctc_greed_search'
    decode_mode = 'ctc_attention_rescoring'
    config = {"sample_rate_hertz": 16000,  'decode_mode': decode_mode}
    audio = {"content": read_pcm(audio)}
    _recognize_request = stt_pb2.RecognizeRequest(config=config, audio=audio)
    response = stub.Recognize(_recognize_request)
    end_time = time.perf_counter()
    cost_time = (end_time - start_time) * 1000
    print("Cost time is: {:2f}".format(cost_time))
    print('*'*20, '解码结果' )
    print(response.results[0].alternatives[0].transcript)





