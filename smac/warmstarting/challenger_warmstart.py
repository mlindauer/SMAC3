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
                             runhist_fn_dict: typing.Dict[str, typing.List[str]],
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
            runhist_fn_dict:typing.Dict[str, typing.List[str]]
                dictionary of scenario file to list of runhistory files
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


        # using EPM, select a subset of initial configs
        if runhist_fn_dict:
            
            rh = RunHistory(aggregate_func=average_cost)
            
            runhist_fn_list = []
            scenario_fn_list = []
            for scen_fn, rh_list in runhist_fn_dict.items():
                scenario_fn_list.extend([scen_fn]*len(rh_list))
                runhist_fn_list.extend(rh_list)
            
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
            model.train(X, y)

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

            if scenario.run_obj == "runtime":
                Y = 10**Y

            # greedy forward selection
            perc_marg_contr = 1
            #sbs_index = self._get_sbs_index(Y)

            # ensure that user default is part of initial design
            initial_configs = [configs[0]]
            configs.remove(configs[0])
            Y_sel = np.array(Y[0, :])
            Y_left = np.delete(Y, 0, axis=1)
            self.logger.info("ADD user default to initial design")

            #self.logger.info("SBS index: %d;" %(sbs_index))

            while True:
                if not configs:
                    break
                vbs = self._get_vbs(Y_sel)
                marg_contr = []
                for i, c in enumerate(configs):
                    Y_ = Y_sel[:, :]
                    Y_ = np.vstack([Y_, Y[i, :]])
                    marg = vbs - self._get_vbs(Y_)
                    marg_contr.append(marg)
                max_marg_index = np.argmax(marg_contr)
                marg = marg_contr[max_marg_index]
                if marg / np.abs(vbs) < self.VBS_THRESHOLD:
                    self.logger.info(
                        "Marginal contribution too small (%f) -- don't adding further initial configurations" % (marg / np.abs(vbs)))
                    break
                self.logger.info("%d : %f (%.2f perc)" %
                                 (max_marg_index, marg, marg / vbs))
                initial_configs.append(configs[max_marg_index])
                configs.remove(configs[max_marg_index])
                Y_sel = np.vstack([Y_sel, Y[i, :]])
                Y_left = np.delete(Y_left, max_marg_index, axis=1)

            return initial_configs

    def _get_vbs(self, Y):
        return np.average(np.min(Y, axis=0))

    def _get_sbs_index(self, Y):
        return np.argmin(np.average(Y, axis=1))
