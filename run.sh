#!/bin/bash
# 실행 스크립트 - 터미널에서 ./run.sh로 실행하세요

cd "$(dirname "$0")"
source venv/bin/activate
python -u main.py

