# -*- coding: utf-8 -*-
from watson_developer_cloud import SpeechToTextV1
from os.path import join, dirname
import sys
import json

speech_to_text = SpeechToTextV1(
    username='1481a417-9ef2-4724-83d1-46ba7b3da959',
    password='vqmkyQbWGlfz',
    url='https://stream.watsonplatform.net/speech-to-text/api'
)

if __name__ == '__main__':
    if len(sys.argv) < 2 :
        print("ERROR! No wavefile input!\n\tUsage : python etri_stt.py [wavefile_path]")

    with open(join(dirname(__file__), './.', sys.argv[1]), 'rb') as audio_file:
        speech_recognition_results = speech_to_text.recognize(
            audio=audio_file,
            content_type='audio/wav',
            timestamps=True,
            model='ko-KR_BroadbandModel',
            word_alternatives_threshold=0.9,
            keywords=[' '],
            keywords_threshold=0.5,
            max_alternatives=3).get_result()

    parse_str = json.loads(json.dumps(speech_recognition_results, indent=2))
    print parse_str["results"][0]["alternatives"][0]["transcript"]
