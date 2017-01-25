import os
import sys
import logging
import numpy as np

from smac.utils.io.cmd_reader import CMDReader
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC
from smac.facade.roar_facade import ROAR
from smac.runhistory.runhistory import RunHistory
from smac.smbo.objective import average_cost
from smac.utils.merge_foreign_data import merge_foreign_data_from_file
from smac.utils.io.traj_logging import TrajLogger
from smac.warmstarting.challenger_warmstart import ChallengerWarmstart

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2017, ML4AAD"
__license__ = "3-clause BSD"


class SMACCLI(object):

    '''
    main class of SMAC
    '''

    def __init__(self):
        '''
            constructor
        '''
        self.logger = logging.getLogger("SMAC")

    def main_cli(self):
        '''
            main function of SMAC for CLI interface
        '''

        cmd_reader = CMDReader()
        args_, misc_args = cmd_reader.read_cmd()

        logging.basicConfig(level=args_.verbose_level)

        root_logger = logging.getLogger()
        root_logger.setLevel(args_.verbose_level)

        scen = Scenario(args_.scenario_file, misc_args)

        if args_.modus == "SMAC":
            optimizer = SMAC(
                scenario=scen,
                rng=np.random.RandomState(args_.seed))
        elif args_.modus == "ROAR":
            optimizer = ROAR(
                scenario=scen,
                rng=np.random.RandomState(args_.seed))

        if args_.warmstart_incumbent:
            optimizer.warmstart_challengers(warmstart_trajectory_fns=args_.warmstart_incumbent,
                                            warmstart_runhistory_fns=args_.warmstart_runhistory,
                                            warmstart_scenario_fns=args_.warmstart_scenario)
        if args_.warmstart_runhistory:
            optimizer.warmstart_model(warmstart_runhistory_fns=args_.warmstart_runhistory,
                                      warmstart_scenario_fns=args_.warmstart_scenario,
                                      warmstart_mode=args_.warmstart_mode)
        try:
            optimizer.optimize()

        finally:
            # ensure that the runhistory is always dumped in the end
            if scen.output_dir is not None:
                optimizer.solver.runhistory.save_json(
                    fn=os.path.join(scen.output_dir, "runhistory.json"))
        #smbo.runhistory.load_json(fn="runhistory.json", cs=smbo.config_space)
