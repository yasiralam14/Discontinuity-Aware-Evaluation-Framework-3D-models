#!/bin/bash

# Define the list of scenes

cd

bash ./renders/automation_scripts/render_radegs_official.sh
bash ./renders/automation_scripts/render_betasplatting.sh

cd

cd ./models/3dcs/convex-splatting/

eval "$(micromamba shell hook --shell bash)"
micromamba activate convex_splatting

bash ./run_all.sh

cd

micromamba deactivate

bash ./renders/automation_scripts/render_3dcs.sh

cd

cd /models/triangle_splatting2/triangle-splatting2/

eval "$(micromamba shell hook --shell bash)"
micromamba activate triangle-splatting2

bash ./tnt_run.sh

cd

micromamba deactivate

bash ./renders/automation_scripts/render_trianglesplatting.sh

eval "$(conda shell.bash hook)"
conda activate linear_splatting

cd
cd ./models/linprim/linear-splatting/
 bash tnt_training.sh