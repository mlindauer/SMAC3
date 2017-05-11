import os
import sys
import logging
import numpy as np

from smac.utils.io.cmd_reader import CMDReader
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC
from smac.facade.roar_facade import ROAR
from smac.runhistory.runhistory import RunHistory
from smac.optimizer.objective import average_cost
from smac.utils.merge_foreign_data import merge_foreign_data_from_file
from smac.utils.io.traj_logging import TrajLogger
from smac.tae.execute_ta_run import TAEAbortException, FirstRunCrashedException

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
        self.logger = logging.getLogger(self.__module__ + "." + self.__class__.__name__)

    def main_cli(self):
        '''
            main function of SMAC for CLI interface
        '''
        self.logger.info("SMAC call: %s" %(" ".join(sys.argv)))

        cmd_reader = CMDReader()
        args_, misc_args = cmd_reader.read_cmd()

        logging.basicConfig(level=args_.verbose_level)

        root_logger = logging.getLogger()
        root_logger.setLevel(args_.verbose_level)

        scen = Scenario(args_.scenario_file, misc_args,
                        run_id=args_.seed)

        if args_.mode == "SMAC":
            optimizer = SMAC(
                scenario=scen,
                rng=np.random.RandomState(args_.seed))
        elif args_.mode == "ROAR":
            optimizer = ROAR(
                scenario=scen,
                rng=np.random.RandomState(args_.seed))

        if args_.warmstart_incumbent:
            optimizer.warmstart_challengers(warmstart_traj_dicts=args_.warmstart_incumbent,
                                            runhist_fn_dict=args_.warmstart_runhistory,
                                            all_insts=args_.wsi_all)
        if args_.warmstart_runhistory:
            optimizer.warmstart_model(runhist_fn_dict=args_.warmstart_runhistory,
                                      warmstart_mode=args_.warmstart_mode)
        try:
            optimizer.optimize()
        except (TAEAbortException, FirstRunCrashedException) as err:
            self.logger.error(err)
