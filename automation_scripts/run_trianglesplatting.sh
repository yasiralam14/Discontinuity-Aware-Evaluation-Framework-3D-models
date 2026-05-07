#!/bin/bash
cd
cd ./automation_scripts
bash train_trianglesplatting2.sh
cd
cd ./renders/automation_scripts
bash ./render_trianglesplatting.sh

cd

cd ./metrics

python metrics.py -m /home/salam4/renders/models/triangle_splatting2/*