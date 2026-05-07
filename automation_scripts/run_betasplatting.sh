#!/bin/bash
cd
cd ./automation_scripts
bash train_betasplatting.sh
cd
cd ./renders/automation_scripts
bash ./render_betasplatting.sh

cd
cd ./metrics

python metrics.py -m /home/salam4/renders/models/beta_splatting/*