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

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2017, ML4AAD"
__license__ = "3-clause BSD"


class ChallengerWarmstart(object):

    def __init__(self, rng):
        '''
            Constructor
        '''
        self.logger = logging.getLogger("ChallengerWarmstart")
        
        self.VBS_THRESHOLD = 0.01
        self.rng = rng

    def get_init_challengers(self,
                             scenario: Scenario,
                             traj_fn_list: typing.List[str],
                             runhist_fn_list: typing.List[str],
                             scenario_fn_list: typing.List[str],
                             hist2epm: AbstractRunHistory2EPM,
                             ):
        '''
            reads provided files and returns a list of previous incumbents as challengers 
            If <runhists> are provided, challengers are filtered by a greedy forward selection to minimize VBS on predicted performance values

            Arguments
            ---------
            scenario: Scenario
                current scenario
            traj_fn_list:typing.List[str]
                list of trajectory files
            runhist_fn_list:typing.List[str]
                list of runhistory files
            scenario_fn_list: typing.List[str]
                list of scenario files (in the same order as runhists
            hist2epm:AbstractRunHistory2EPM
                object to convert runhistories into EPM training data
                
            Returns
            -------
            typing.List[Configuration]
            
        '''

        scenario = copy.deepcopy(scenario)
        initial_configs = None

        initial_configs = [scenario.cs.get_default_configuration()]
        for traj_fn in traj_fn_list:
            trajectory = TrajLogger.read_traj_aclib_format(
                fn=traj_fn, cs=scenario.cs)
            initial_configs.append(trajectory[-1]["incumbent"])

        rh = RunHistory(aggregate_func=average_cost)

        if runhist_fn_list and scenario_fn_list:
            scenario, rh = merge_foreign_data_from_file(
                scenario=scenario, runhistory=rh,
                in_scenario_fn_list=scenario_fn_list,
                in_runhistory_fn_list=runhist_fn_list,
                cs=scenario.cs,
                aggregate_func=average_cost,
                update_train=True)

            # update feature array
            scenario._update_feature_array()
            hist2epm.instance_features = scenario.feature_dict

            # Convert rh
            X, y = hist2epm.transform(runhistory=rh)
            # initial EPM
            types = get_types(scenario.cs, scenario.feature_array)
            model = RandomForestWithInstances(types=types,
                                              instance_features=scenario.feature_array,
                                              seed=self.rng.randint(MAXINT))
            model.train(X,y)
            
            configs = initial_configs[:]
            
            imputed_configs = map(ConfigSpace.util.impute_inactive_values,
                                       configs)
            C = [x.get_array() for x in imputed_configs]
            Y = []
            n_instances = len(scenario.feature_array)
            for c in enumerate(C):
                X_ = np.hstack(
                    (np.tile(c[1], (n_instances, 1)), scenario.feature_array))
                y = model.predict(X_)
                Y.append(y[0])
            Y = np.array(Y)
            
            # greedy forward selection
            perc_marg_contr = 1
            sbs_index = self._get_sbs_index(Y)
            initial_configs = [configs[sbs_index]]
            configs.remove(configs[sbs_index])
            Y = np.delete(Y,sbs_index,axis=1)
            
            self.logger.info("SBS index: %d;" %(sbs_index))
            
            while True:
                if not configs:
                    break
                vbs = self._get_vbs(Y)
                marg_contr = []
                for i,c in enumerate(configs):
                    Y_ = Y[:,:]
                    Y_ = np.delete(Y_,i,axis=1)
                    marg = self._get_vbs(Y_) - vbs
                    marg_contr.append(marg)
                max_marg_index = np.argmax(marg_contr)
                marg = marg_contr[max_marg_index]
                if marg / vbs < self.VBS_THRESHOLD:
                    break
                self.logger.info("%d : %f (%.2f\%)" %(max_marg_index, marg, marg / vbs))
                initial_configs.append(configs[max_marg_index])
                configs.remove(configs[max_marg_index])
                Y = np.delete(Y,max_marg_index,axis=1)
                
            return initial_configs
                    
                
    def _get_vbs(self, Y):
        return np.average(np.min(Y,axis=0))
    
    def _get_sbs_index(self, Y):
        return np.argmin(np.average(Y,axis=1))