#!/usr/bin/env bash
set -e

python -m pip install --upgrade pip setuptools wheel

pip install numpy==1.26.4
pip install scipy==1.10.1 --only-binary=:all:
pip install faiss-cpu==1.7.3 --no-build-isolation
pip install implicit==0.7.2

pip install -r requirements.txt --no-build-isolation
