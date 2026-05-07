#!/bin/bash
cd
cd ./automation_scripts
bash train_3dcs.sh
cd
cd ./renders/automation_scripts
bash ./render_3dcs.sh

cd
cd ./metrics

python metrics.py -m /home/salam4/renders/models/3dcs/*
