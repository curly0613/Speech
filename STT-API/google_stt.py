# -*- coding: utf-8 -*-
import io
import sys
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types

def Google_STT(audio_file, contexts=[' ']) :
    client = speech.SpeechClient()

    with io.open(audio_file, 'rb') as wavefile:
        content = wavefile.read()
        audio = types.RecognitionAudio(content=content)
        config = types.RecognitionConfig(encoding = enums.RecognitionConfig.AudioEncoding.LINEAR16, sample_rate_hertz = 16000, language_code = 'ko-KR', speech_contexts=[speech.types.SpeechContext(phrases=contexts)])
        google_result = client.recognize(config,audio)

    if len(google_result.results) == 0:
        return "empty"

    return google_result.results[0].alternatives[0].transcript.encode('utf-8')


if __name__ == '__main__':
    if len(sys.argv) < 2 :
        print("ERROR! No wavefile input!\n\tUsage : python google_stt.py [wavefile_path]")
    print(Google_STT(sys.argv[1], contexts=[' ']))
