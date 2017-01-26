import typing
import copy
import logging

import numpy as np

import ConfigSpace.util

from smac.runhistory.runhistory2epm import AbstractRunHistory2EPM
from smac.runhistory.runhistory import RunHistory
from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.configspace import ConfigurationSpace
from smac.utils.io.traj_logging import TrajLogger
from smac.utils.merge_foreign_data import merge_foreign_data_from_file
from smac.scenario.scenario import Scenario
from smac.smbo.objective import average_cost
from smac.utils.util_funcs import get_types
from smac.utils.constants import MAXINT
from smac.epm.rf_with_instances_warmstarted import WarmstartedRandomForestWithInstances
from smac.smbo.acquisition import EI, AbstractAcquisitionFunction, WARM_EI

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2017, ML4AAD"
__license__ = "3-clause BSD"


class TransferWarmstart(object):

    def __init__(self, rng: np.random.RandomState):
        '''
            constructor
            
            Arguments
            ---------
            rng: np.random.RandomState
                random state
        '''

        self.logger = logging.getLogger("TransferWarmstart")
        self.rng = rng

    def _prepare(self,
                 warmstart_runhistory_fns: typing.List[str],
                 warmstart_scenario_fns: typing.List[str],
                 runhistory2epm: AbstractRunHistory2EPM,
                 cs: ConfigurationSpace
                 ):
        '''
            read files and learn EPM for each given runhistory

            Arguments
            ---------
            warmstart_runhistory_fns: typing.List[str],
                RunHistory files to initialize 
                WarmstartedRandomForestWithInstances as EPM 
            warmstart_scenario_fns: typing.List[str],
                Scenario files to provide information
                how to interpret warmstart_runhistories;
                has to have the same length as warmstart_runhistories
            runhistory2epm : AbstractRunHistory2EPM
                object to convert runhistory to X,y for EPM training
            cs: ConfigurationSpace
                parameter configuration space
        '''
        aggregate_func = average_cost

        warmstart_runhistories = []
        warm_scenarios = []
        if len(warmstart_runhistory_fns) != len(warmstart_scenario_fns):
            raise ValueError(
                "warmstart_runhistory and warmstart_scenario have to have the same lengths")

        # read files
        for rh_fn in warmstart_runhistory_fns:
            warm_rh = RunHistory(aggregate_func=aggregate_func)
            warm_rh.load_json(fn=rh_fn, cs=cs)
            warmstart_runhistories.append(warm_rh)
        for warm_scen in warmstart_scenario_fns:
            warm_scenarios.append(
                Scenario(scenario=warm_scen, cmd_args={"output_dir": ""}))

        warmstart_models = []
        
        runhistory2epm = copy.copy(runhistory2epm)
        for rh, warm_scen in zip(warmstart_runhistories, warm_scenarios):
            if not rh.data:  # skip empty rh
                continue
            # patch runhistory to use warmstart_scenario
            runhistory2epm.scenario = warm_scen
            runhistory2epm.instance_features = warm_scen.feature_dict
            runhistory2epm.n_feats = warm_scen.n_features
            X, y = runhistory2epm.transform(rh)
            warm_types = get_types(warm_scen.cs, warm_scen.feature_array)
            warm_model = RandomForestWithInstances(types=warm_types,
                                                   instance_features=warm_scen.feature_array,
                                                   seed=self.rng.randint(MAXINT))
            warm_model.train(X, y)
            warmstart_models.append(warm_model)

        return warmstart_models

    def get_warmstart_EPM(self,
                          scenario: Scenario,
                          warmstart_runhistory_fns: typing.List[str],
                          warmstart_scenario_fns: typing.List[str],
                          runhistory2epm: AbstractRunHistory2EPM):
        '''
            initializes an EPM which weights the current EPM 
            and EPMs from each given runhistory

            Arguments
            ---------
            scenario: Scenario
                AC scenario
            warmstart_runhistory_fns: typing.List[str],
                RunHistory files to initialize 
                WarmstartedRandomForestWithInstances as EPM 
            warmstart_scenario_fns: typing.List[str],
                Scenario files to provide information
                how to interpret warmstart_runhistories;
                has to have the same length as warmstart_runhistories
            runhistory2epm : AbstractRunHistory2EPM
                object to convert runhistory to X,y for EPM training

            Returns
            -------
            WarmstartedRandomForestWithInstances
        '''
        self.logger.info("Use \"weighted\" warmstart strategy")

        warmstart_models = self._prepare(
            warmstart_runhistory_fns, warmstart_scenario_fns, runhistory2epm,
            cs=scenario.cs)
        types = get_types(scenario.cs, scenario.feature_array)

        return WarmstartedRandomForestWithInstances(types=types,
                                                    instance_features=scenario.feature_array,
                                                    seed=self.rng.randint(
                                                        MAXINT),
                                                    warmstart_models=warmstart_models)

    def get_WARM_EI(self,
                    scenario: Scenario,
                    model: RandomForestWithInstances,
                    warmstart_runhistory_fns: typing.List[str],
                    warmstart_scenario_fns: typing.List[str],
                    runhistory2epm: AbstractRunHistory2EPM):
        '''
            trains a transfer function which is added to the acquisition function

            Arguments
            ---------
            scenario: Scenario
                AC scenario
            warmstart_runhistory_fns: typing.List[str],
                RunHistory files to initialize 
                WarmstartedRandomForestWithInstances as EPM 
            warmstart_scenario_fns: typing.List[str],
                Scenario files to provide information
                how to interpret warmstart_runhistories;
                has to have the same length as warmstart_runhistories

            Returns
            -------
            WARM_EI
        '''
        self.logger.info("Use \"transfer\" warmstart strategy")

        warmstart_models = self._prepare(
            warmstart_runhistory_fns, warmstart_scenario_fns, runhistory2epm,
            cs=scenario.cs)
        
        for w_model in warmstart_models:
            w_model.train_norm()

        return WARM_EI(model=model, warm_models=warmstart_models)
