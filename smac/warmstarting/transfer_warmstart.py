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
                 runhist_fn_dict:typing.Dict[str, typing.List[str]],
                 runhistory2epm: AbstractRunHistory2EPM,
                 cs: ConfigurationSpace
                 ):
        '''
            read files and learn EPM for each given runhistory

            Arguments
            ---------
            runhist_fn_dict:typing.Dict[str, typing.List[str]]
                dictionary of scenario file to list of runhistory files
            runhistory2epm : AbstractRunHistory2EPM
                object to convert runhistory to X,y for EPM training
            cs: ConfigurationSpace
                parameter configuration space
        '''
        aggregate_func = average_cost

        warmstart_models = []
        runhistory2epm = copy.copy(runhistory2epm)
        for scen_fn, rh_fn_list in runhist_fn_dict.items():
            warm_scen = Scenario(scenario=scen_fn, cmd_args={"output_dir": ""})
            warm_rh = RunHistory(aggregate_func=aggregate_func)
            
            warm_scen, warm_rh = merge_foreign_data_from_file(
                scenario=warm_scen, runhistory=warm_rh,
                in_scenario_fn_list=[scen_fn]*len(rh_fn_list),
                in_runhistory_fn_list=rh_fn_list,
                cs=warm_scen.cs,
                aggregate_func=average_cost,
                update_train=True)
            
            if not warm_rh.data:  # skip empty rh
                continue
            # patch runhistory to use warmstart_scenario
            runhistory2epm.scenario = warm_scen
            runhistory2epm.instance_features = warm_scen.feature_dict
            runhistory2epm.n_feats = warm_scen.n_features
            X, y = runhistory2epm.transform(warm_rh)
            warm_types = get_types(warm_scen.cs, warm_scen.feature_array)
            warm_model = RandomForestWithInstances(types=warm_types,
                                                   instance_features=warm_scen.feature_array,
                                                   seed=self.rng.randint(MAXINT))
            warm_model.train(X, y)
            warmstart_models.append(warm_model)

        return warmstart_models

    def get_warmstart_EPM(self,
                          scenario: Scenario,
                          runhist_fn_dict:typing.Dict[str, typing.List[str]],
                          runhistory2epm: AbstractRunHistory2EPM):
        '''
            initializes an EPM which weights the current EPM 
            and EPMs from each given runhistory

            Arguments
            ---------
            scenario: Scenario
                AC scenario
            runhist_fn_dict:typing.Dict[str, typing.List[str]]
                dictionary of scenario file to list of runhistory files
            runhistory2epm : AbstractRunHistory2EPM
                object to convert runhistory to X,y for EPM training

            Returns
            -------
            WarmstartedRandomForestWithInstances
        '''
        self.logger.info("Use \"weighted\" warmstart strategy")

        warmstart_models = self._prepare(
            runhist_fn_dict=runhist_fn_dict, 
            runhistory2epm=runhistory2epm,
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
                    runhist_fn_dict:typing.Dict[str, typing.List[str]],
                    runhistory2epm: AbstractRunHistory2EPM):
        '''
            trains a transfer function which is added to the acquisition function

            Arguments
            ---------
            scenario: Scenario
                AC scenario
            model: RandomForestWithInstances
                EPM model to be used for predictions on new data
            runhist_fn_dict:typing.Dict[str, typing.List[str]]
                dictionary of scenario file to list of runhistory files

            Returns
            -------
            WARM_EI
        '''
        self.logger.info("Use \"transfer\" warmstart strategy")

        warmstart_models = self._prepare(
            runhist_fn_dict=runhist_fn_dict, 
            runhistory2epm=runhistory2epm,
            cs=scenario.cs)
        
        for w_model in warmstart_models:
            w_model.train_norm()

        return WARM_EI(model=model, warm_models=warmstart_models)
