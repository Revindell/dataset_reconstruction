# --------------------------------------------------------------------------------
# Note: Requires https://gitlab.lrz.de/cps/commonroad-geometric to be installed
# --------------------------------------------------------------------------------

from commonroad_geometric.dataset.iteration.scenario_iterator import ScenarioIterator
from commonroad_geometric.rendering.video_recording import render_scenario_movie

if __name__ == '__main__':
    for scenario_bundle in ScenarioIterator('./data/argoverse'):
        render_scenario_movie(scenario_bundle.input_scenario)

    # TODO @Luis: Add your data here  
    # for scenario in ScenarioIterator('./data/XXXXXXXX'):
    #     render_scenario_movie(scenario)
