import typing
import copy
import logging

import numpy as np

import ConfigSpace.util

from smac.runhistory.runhistory2epm import AbstractRunHistory2EPM
from smac.runhistory.runhistory import RunHistory
from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.configspace import ConfigurationSpace, convert_configurations_to_array
from smac.utils.io.traj_logging import TrajLogger
from smac.utils.merge_foreign_data import merge_foreign_data_from_file
from smac.scenario.scenario import Scenario
from smac.optimizer.objective import average_cost
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

        self.MARG_THRESHOLD = 0.01
        self.rng = rng

    def get_init_challengers(self,
                             scenario: Scenario,
                             traj_dicts: typing.Dict[str, typing.List[str]],
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
            traj_dicts: typing.Dict[str, typing.List[str]]
                dictionary of scenario file to list of trajectory files
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
        for scen_name, traj_fns in traj_dicts.items():
            for traj_fn in traj_fns:
                trajectory = TrajLogger.read_traj_aclib_format(
                    fn=traj_fn, cs=scenario.cs)
                inc = trajectory[-1]["incumbent"]
                inc.origin = scen_name
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
            C = convert_configurations_to_array(configs)
            
            # get predictions for each configurations on each instance
            Y = []
            n_instances = len(scenario.feature_array)
            for c in enumerate(C):
                X_ = np.hstack(
                    (np.tile(c[1], (n_instances, 1)), scenario.feature_array))
                y = model.predict(X_)
                Y.append(np.ravel(y[0]))
            Y = np.array(Y)
            
            if scenario.run_obj == "runtime":
                Y = 10**Y

            # ensure that user default is part of initial design
            sel_configs = [configs[0]]
            sel_Y = [Y[0]]
            # select one config per scenario
            for origin in traj_dicts.keys():
                scores = []
                for c,Y_ in zip(configs,Y):
                    if c.origin == origin:
                        scores.append((np.mean(Y_),Y_,c))
                best_c_indx = np.argmin([s[0] for s in scores])
                self.logger.debug("Best predicted cost %.2f for origin %s" %(scores[best_c_indx][0], origin))
                sel_Y.append(scores[best_c_indx][1])
                sel_configs.append(scores[best_c_indx][2])
            configs = sel_configs
            Y = np.array(sel_Y)

            # greedy forward selection
            perc_marg_contr = 1
            #sbs_index = self._get_sbs_index(Y)

            # again ensure that user default is part of initial design
            initial_configs = [configs[0]]
            configs.remove(configs[0])
            Y_sel = np.reshape(np.array(Y[0, :]), (1,Y.shape[1]))
            Y_left = np.delete(Y, 0, axis=0)
            self.logger.info("ADD user default to initial design")

            #self.logger.info("SBS index: %d;" %(sbs_index))

            while True:
                if not configs:
                    break
                vbs = self._get_vbs(Y_sel)
                marg_contr = []
                for i, c in enumerate(configs):
                    Y_add = np.reshape(Y_left[i, :], (1,Y.shape[1]))
                    Y_ = np.vstack([Y_sel, Y_add])
                    marg_impr = 1 - (self._get_vbs(Y_) / vbs)
                    marg_contr.append(marg_impr)
                max_marg_index = np.argmax(marg_contr)
                marg = marg_contr[max_marg_index]
                if marg < self.MARG_THRESHOLD:
                    self.logger.info(
                        "Marginal contribution improvement too small (%f%%) -- don't adding further initial configurations" % (marg))
                    break
                self.logger.info("%d : %f%%" %
                                 (max_marg_index, marg))
                initial_configs.append(configs[max_marg_index])
                configs.remove(configs[max_marg_index])
                Y_add = np.reshape(Y_left[i, :], (1,Y.shape[1]))
                Y_sel = np.vstack([Y_sel, Y_add])
                Y_left = np.delete(Y_left, max_marg_index, axis=0)

        return initial_configs
        

    def _get_vbs(self, Y):
        return np.average(np.min(Y, axis=0))

    def _get_sbs_index(self, Y):
        return np.argmin(np.average(Y, axis=1))
