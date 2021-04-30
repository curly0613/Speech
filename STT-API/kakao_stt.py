# -*- coding: utf-8 -*-
import os
import sys
import json
import subprocess

Authorization = "828f819d425933148351cfdb3c87e79e"

if __name__ == '__main__':
    if len(sys.argv) < 2 :
        print("ERROR! No wavefile input!\n\tUsage : python etri_stt.py [wavefile_path]")

    audio_path = sys.argv[1]

    with open(os.devnull, 'w') as devnull:
        result = subprocess.check_output(["curl", "-v", "-X", "POST", "https://kakaoi-newtone-openapi.kakao.com/v1/recognize", "-H", "Transfer-Encoding:chunked", "-H", "Content-Type:application/octet-stream", "-H", "X-DSS-Service:DICTATION", "-H", "Authorization:KakaoAK "+Authorization, "--data-binary", "@"+audio_path])
    print(json.loads(result.split('\n')[-3])["value"])
