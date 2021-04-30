# -*- coding: utf-8 -*-
import sys
import urllib3
import json
import base64

openApiURL = "http://aiopen.etri.re.kr:8000/WiseASR/Recognition"
accessKey = "aa63c4e5-f891-4580-ba91-cf8ca689c202"
languageCode = "korean"

if __name__ == '__main__':
    if len(sys.argv) < 2 :
        print("ERROR! No wavefile input!\n\tUsage : python etri_stt.py [wavefile_path]")

    file = open(sys.argv[1], "rb")
    audioContents = base64.b64encode(file.read()).decode("utf8")
    file.close()

    requestJson = {
        "access_key": accessKey,
        "argument": {
            "language_code": languageCode,
            "audio": audioContents
        }
    }

    http = urllib3.PoolManager()
    response = http.request(
        "POST",
        openApiURL,
        headers={"Content-Type": "application/json; charset=UTF-8"},
        body=json.dumps(requestJson)
    )
    print(json.loads(response.data)["return_object"]["recognized"])
