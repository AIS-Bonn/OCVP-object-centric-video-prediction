#!/usr/bin/env bash
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1XSLW3qBtcxxvV-5oiRruVTlDlQ_Yatzm' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1XSLW3qBtcxxvV-5oiRruVTlDlQ_Yatzm" -O datasets/OBJ3D.zip && rm -rf /tmp/cookies.txt
cd datasets && unzip OBJ3D.zip && rm OBJ3D.zip
