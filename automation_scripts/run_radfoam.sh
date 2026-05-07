#!/bin/bash
cd
cd ./automation_scripts
bash train_radfoam.sh
cd
cd ./renders/automation_scripts
bash ./render_radfoam.sh

cd

cd ./metrics

python metrics.py -m /home/salam4/renders/models/radfoam/*