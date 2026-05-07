#!/bin/bash
cd
cd ./automation_scripts

bash run_linprim.sh
bash run_betasplatting.sh
bash run_radfoam.sh
bash run_3dcs.sh
bash run_QGS.sh
bash run_trianglesplatting.sh

cd
cd ./metrics

bash run_metrics.sh

cd
