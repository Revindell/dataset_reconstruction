from pathlib import Path

from scenario_cleaner import StrategyChoices
from scenario_cleaner import main

# essentially just calls the UI and passes the arguments
# useful for debugging or trying out different settings

main(Path('data/argoverse/USA_ArgoverseMIA-10_0_T-1.xml'),  # input path
     Path('cleaneddata/argoverse/'),  # output path
     strategy=[
         StrategyChoices.offroad,
         StrategyChoices.rotate,
         StrategyChoices.traj_noise,
         StrategyChoices.static,
         StrategyChoices.obj_loss,
         StrategyChoices.duplicates,
     ])

# USA_ArgoverseMIA-4_0_T-1.xml
# /home/deyu/dataset-reconstruction/data/argoverse/USA_ArgoverseMIA-144_0_T-1.xml
# USA_ArgoversePIT-106_0_T-1.xml

# USA_ArgoverseMIA-10_0_T-1.xml
# USA_ArgoverseMIA-54_0_T-1.xml
