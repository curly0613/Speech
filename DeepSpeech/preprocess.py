import wave
import contextlib
import json
import csv
import os

def Calduration(wavepath) :
    with contextlib.closing(wave.open(wavepath,'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
    return duration

def char_to_code(char):
  code = ord(char) - 0xAC00
  jong = code % 28
  jung = (code-jong)//28%21
  cho = ((code-jong)//28-jung)//21
  if cho >= 19 or cho < 0:
    print("cannot encode:", cho)
  return cho+1, jung+19, jong+39

def Kor_split(text) :
    out = ''
    for i in text :
        if i == ' ' :
            out += ' '
        else :
            cho, jung, jong = char_to_code(i)
            if not cho == 12 :
                if cho > 12 :
                    cho -= 1
                out += map_tab[cho]
            out += map_tab[jung]
            if not jong == 39 :
                out += map_tab[jong]
    return out

if __name__ == '__main__':
    f_lst = []
    with open('data/corpus_trim.csv', 'r') as csvfile:
        wt = csv.reader(csvfile, quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in wt :
            f_lst.append(i)
    f_lst = f_lst[1:]

    map_tab = []
    with open("data/hunmin_lm.txt", "r") as file:
        for i in file :
            map_tab.append(i[:-1])
    map_tab = map_tab[4:]

    entry_lst = []
    for i in f_lst :
        entry = {'duration': Calduration(i[0]), 'text': Kor_split(i[2].decode('utf-8')), 'key': "/data3/dongk/DeepSpeech/"+i[0]}

        entry_lst.append(entry)

    with open("data/train_list.json", 'w') as file:
        for i in entry_lst :
            file.write(json.dumps(i))
            file.write('\n')
