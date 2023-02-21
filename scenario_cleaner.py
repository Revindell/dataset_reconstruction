import os
from enum import Enum
from pathlib import Path
from typing import List, Optional, Any
import multiprocessing as mp
from tqdm import tqdm

import typer
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.common.file_writer import CommonRoadFileWriter, OverwriteExistingFile

from clean_duplicates import CleanDuplicates
from clean_offroad_detection import CleanOffroad
from clean_rotating import CleanRotating
from clean_static import CleanStatic
from clean_traj_noise import CleanTrajNoise, KFModelChoices
from clean_object_loss import CleanObjectLoss
from strategy_interface import *
import istarmap


# Context class
class Cleaner:
    """ Main class of the data cleaner. Interface to the client.
    """

    def __init__(self, strategy: Strategy) -> None:
        """ Constructor of class Cleaner.

        :param strategy: Cleaning algorithm to be applied
        """
        self._strategy = strategy

    @property
    def strategy(self) -> Strategy:
        """ Defines the used strategy.

        :getter: returns the strategy/cleaning algorithm
        :setter: sets the strategy/cleaning algorithm
        :type: Strategy class instance
        """
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: Strategy) -> None:
        self._strategy = strategy

    def clean_data(self, scenario, **kwargs) -> Any:
        """ Calls the cleaning algorithm.

        :param scenario: scenario to be changed
        :return: new scenario
        """
        return self._strategy.clean_scenario(scenario, **kwargs)


class StrategyChoices(str, Enum):
    """ Class which describes the predefined choices for the CLI.
    """
    static = 'static'
    rotate = 'rotate'
    offroad = 'offroad'
    obj_loss = 'obj_loss'
    traj_noise = 'traj_noise'
    duplicates = 'duplicates'
    all = 'all'


def main(input_dir: Path,
         output_dir: Path,
         strategy: Optional[List[StrategyChoices]] = typer.Option(None,
                                                                  help='Multiple cleaning algorithms can be chosen.')) \
        -> None:
    """ Reads the scenario file, applies the desired algorithms and outputs new scenario file. Describes CLI.

    :param input_dir: directory of files or single file which are to be processed
    :param output_dir: directory where new files are saved
    :param strategy: list of applied cleaning algorithms
    :return: None
    """
    if not strategy:
        typer.echo('No cleaning algorithms chosen')
        raise typer.Abort()
    if input_dir.is_file():
        scenario_paths = [input_dir]
    else:
        scenario_paths = list(input_dir.rglob('*.xml'))
    # get number of processors
    num_processors = mp.cpu_count()
    # parallel processing requires iterative arguments. Create new arguments
    list_inputs = []
    for scenario_file in scenario_paths:
        list_inputs.append([scenario_file, strategy, output_dir])
    # pool.starmap(transform_scenario, list_inputs)
    with mp.Pool(num_processors) as pool:
        for _ in tqdm(pool.istarmap(transform_scenario, list_inputs), total=len(list_inputs)):
            pass
    typer.echo('\nStatic obstacles algorithm was applied!') if 'static' in strategy or 'all' in strategy else None
    typer.echo('\nRotating objects algorithm was applied!') if 'rotate' in strategy or 'all' in strategy else None
    typer.echo('\nOff-road detections algorithm was applied!') if 'offroad' in strategy or 'all' in strategy else None
    typer.echo('\nObject loss algorithm was applied!') if 'obj_loss' in strategy or 'all' in strategy else None
    typer.echo('\nTrajectory noise algorithm was applied!') if 'traj_noise' in strategy or 'all' in strategy else None
    typer.echo('\nOverlapping duplicates algorithm was applied!') if 'duplicates' in strategy or 'all' in strategy \
        else None


def transform_scenario(scenario_file: Path, strategy: List[StrategyChoices], output_dir: Path) -> None:
    """ Iterative function to be used in the starmap function for parallel progressing.

    :param scenario_file: file of a scenario
    :param strategy: strategies to be applied
    :param output_dir: output directory
    :return: -
    """
    scenario, planning_problem_set = CommonRoadFileReader(str(scenario_file)).open()
    # print(str(scenario_file))
    # count number of obstacles before cleaning
    num_obstacles_before = len(scenario.obstacles)
    for strat in strategy:
        if strat == 'offroad' or strat == 'all':
            cleaner = Cleaner(CleanOffroad())
            scenario = cleaner.clean_data(scenario, use_collision_check=False)
            # :param use_collision_check: if True, use collision check with road boundary additionally
        if strat == 'rotate' or strat == 'all':
            cleaner = Cleaner(CleanRotating())
            scenario = cleaner.clean_data(scenario)
        if strat == 'static' or strat == 'all':
            cleaner = Cleaner(CleanStatic())
            scenario = cleaner.clean_data(scenario, roadside_parking_check=False, remove_all=False)
            # :param roadside_parking_check: if True, the scenario is checked for roadside parking
        if strat == 'obj_loss' or strat == 'all':
            cleaner = Cleaner(CleanObjectLoss())
            scenario = cleaner.clean_data(scenario)
        if strat == 'traj_noise' or strat == 'all':
            cleaner = Cleaner(CleanTrajNoise())
            scenario = cleaner.clean_data(scenario, model_choice=None, measurement_noise=None, process_noise=None)
            # scenario = cleaner.clean_data(scenario, model_choice=[KFModelChoices.EKFConstAccelerationConstTurnRate, KFModelChoices.EKFConstVelocityConstYawRate, KFModelChoices.EKFPointMass], measurement_noise=None, process_noise=None)
            # :param model_choice: choice of the kalmann filter model to be used, if None, the default model EKFConstAccelerationConstTurnRate is used
            # :param measurement_noise: measurement noise of the kalmann filter, if None, the default value is used
            # :param process_noise: process noise of the kalmann filter, if None, the default value is used
        if strat == 'duplicates' or strat == 'all':
            cleaner = Cleaner(CleanDuplicates())
            scenario = cleaner.clean_data(scenario)

    # count number of obstacles after cleaning
    num_obstacles_after = len(scenario.obstacles)
    # calculate the number of removed obstacles
    print(f'{num_obstacles_before} obstacles before cleaning, {num_obstacles_after} after cleaning,'
          f'{num_obstacles_before - num_obstacles_after} removed in total')
    with open('docs/total_removal_stats.txt', 'a') as f:
        f.write(f'In scenario {scenario.scenario_id}, {num_obstacles_before} obstacles before cleaning,'
                f'{num_obstacles_after} after cleaning, {num_obstacles_before - num_obstacles_after}'
                f'removed in total\n')
    f.close()
    # write new scenario file
    fw = CommonRoadFileWriter(scenario, planning_problem_set)
    filename = os.path.basename(scenario_file)
    output_path = str(output_dir) + "/" + filename
    # print(output_path)
    fw.write_to_file(output_path, OverwriteExistingFile.ALWAYS)


if __name__ == '__main__':
    typer.run(main)
