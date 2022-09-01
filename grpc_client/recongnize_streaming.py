import stt_pb2_grpc
import stt_pb2
import grpc
import time
#from scipy.io.wavfile import read
import argparse
#import numpy as np
#from vad_fun import read_wave, write_wave, webrtcvad, frame_generator, vad_collector

def read_pcm(audio):
    with open(audio,'rb') as f:
        data = f.read()
    return data

def iter_fun(audio, sample_rate):
    config = {"sample_rate_hertz": sample_rate}
    print("Starting...")
    chunksize = int( sample_rate * 0.3)
    with open(audio, 'rb') as f:
        audio_content = f.read()
    print('音频总时长：', len(audio_content))
    start = 0
    end=0
    frames = []
    while start < len(audio_content):
        if len(frames)==0:
            _recognize_request = stt_pb2.StreamingRecognizeRequest(config=config)
            frames.append('flag')
            #frame+='000'
        else:
            if end>len(audio_content): 
                end = len(audio_content)
            else:
                end = start + chunksize
            chunk_audio = audio_content[start:end]
            #print(len(chunk_audio))
            start = end
            #frames.append(chunk_audio)
            #frames+=chunk_audio
            _recognize_request = stt_pb2.StreamingRecognizeRequest(audio_content = chunk_audio)
        yield _recognize_request    
    #with open('t.wav', 'w') as f2:
    #    f2.write(frames[:])
    _recognize_request = stt_pb2.StreamingRecognizeRequest(if_end = True)
    yield _recognize_request    

 
if __name__ == '__main__':
    
    #audio = 'test_audio/'+'BAC009S0764W0136.wav'
    audio = 'test_audio/'+'DEV1.wav'
    #audio = 'test_audio/'+'DEV1_no_end.wav'
    #audio = 'test_audio/'+'DEV1_1.wav'
    #audio = 'test_audio/'+'hua.wav'
    #audio = 'test_audio/ASR/'+'192.wav'
    #audio = 'exp2/'+'D4-spk0-0.wav'
    channel = grpc.insecure_channel("192.168.0.88:5512")
    stub = stt_pb2_grpc.STTStub(channel)
    sample_rate = 16000
    ts = time.time()
    response = stub.StreamingRecognize( iter_fun(audio, sample_rate ) )
    for msg in response:
        res = msg.results[0].alternatives[0].transcript
        print(res)
    te = time.time()
    print(te-ts)
