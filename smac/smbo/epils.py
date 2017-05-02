import itertools
import logging
import numpy as np
import random
import time
import typing
import math


from smac.smbo.acquisition import AbstractAcquisitionFunction
from smac.smbo.base_solver import BaseSolver
from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.smbo.local_search import LocalSearch
from smac.intensification.intensification import Intensifier
from smac.smbo import pSMAC
from smac.runhistory.runhistory import RunHistory
from smac.runhistory.runhistory2epm import AbstractRunHistory2EPM
from smac.stats.stats import Stats
from smac.initial_design.initial_design import InitialDesign
from smac.scenario.scenario import Scenario
from smac.configspace import Configuration, convert_configurations_to_array, get_one_exchange_neighbourhood
from smac.tae.execute_ta_run import TAEAbortException, BudgetExhaustedException
from smac.tae.execute_ta_run import FirstRunCrashedException


__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2017, ML4AAD"
__license__ = "3-clause BSD"


class EPILS_Solver(BaseSolver):

    def __init__(self,
                 scenario: Scenario,
                 stats: Stats,
                 initial_design: InitialDesign,
                 runhistory: RunHistory,
                 runhistory2epm: AbstractRunHistory2EPM,
                 intensifier: Intensifier,
                 aggregate_func: callable,
                 num_run: int,
                 model: RandomForestWithInstances,
                 acq_optimizer: LocalSearch,
                 acquisition_func: AbstractAcquisitionFunction,
                 rng: np.random.RandomState):
        '''
        Interface that contains the main Bayesian optimization loop

        Parameters
        ----------
        scenario: smac.scenario.scenario.Scenario
            Scenario object
        stats: Stats
            statistics object with configuration budgets
        initial_design: InitialDesign
            initial sampling design
        runhistory: RunHistory
            runhistory with all runs so far
        runhistory2epm : AbstractRunHistory2EPM
            Object that implements the AbstractRunHistory2EPM to convert runhistory data into EPM data
        intensifier: Intensifier
            intensification of new challengers against incumbent configuration (probably with some kind of racing on the instances)
        aggregate_func: callable
            how to aggregate the runs in the runhistory to get the performance of a configuration
        num_run: int
            id of this run (used for pSMAC)
        model: RandomForestWithInstances
            empirical performance model (right now, we support only RandomForestWithInstances)
        acq_optimizer: LocalSearch
            optimizer on acquisition function (right now, we support only a local search)
        acquisition_function : AcquisitionFunction
            Object that implements the AbstractAcquisitionFunction (i.e., infill criterion for acq_optimizer)
        rng: np.random.RandomState
            Random number generator
        '''
        self.logger = logging.getLogger(self.__module__ + "." + self.__class__.__name__)
        self.incumbent = None
        self._local_inc = None

        self.scenario = scenario
        self.config_space = scenario.cs
        self.stats = stats
        self.initial_design = initial_design
        self.runhistory = runhistory
        self.rh2EPM = runhistory2epm
        self.intensifier = intensifier
        self.aggregate_func = aggregate_func
        self.num_run = num_run
        self.model = model
        self.acq_optimizer = acq_optimizer
        self.acquisition_func = acquisition_func
        self.rng = rng
        
        self.RESTART_PROB = 0.01
        self.MAX_NEIGHBORS = 5
        self.PERTUBATION_N = 3
        
    def run(self):
        '''
        Runs the Bayesian optimization loop

        Returns
        ----------
        incumbent: np.array(1, H)
            The best found configuration
        '''
        self.stats.start_timing()
        try:
            self.incumbent = self.initial_design.run()
        except FirstRunCrashedException as err:
            if self.scenario.abort_on_first_run_crash:
                raise

        # Main loop
        iteration = 1
        while True:
            if self.scenario.shared_model:
                pSMAC.read(run_history=self.runhistory,
                           output_dirs=self.scenario.input_psmac_dirs,
                           configuration_space=self.config_space,
                           logger=self.logger)

            # model training
            self.logger.info("Model Training")
            X, Y = self.rh2EPM.transform(self.runhistory)
            self.model.train(X, Y)            
            self.acquisition_func.update(model=self.model, eta=self.runhistory.get_cost(self.incumbent))

            if iteration == 1:
                start_point = self.incumbent
            else:
                # Restart?
                if random.random() < self.RESTART_PROB:
                    self.logger.info("Restart Search")
                    start_point = self.scenario.cs.sample_configuration()
                else:
                    # pertubate inc
                    self.logger.info("Pertubate Incumbent")
                    start_point = self.incumbent
                    for _ in range(self.PERTUBATION_N):
                        start_point = random.choice(list(get_one_exchange_neighbourhood(
                            start_point, seed=self.rng.seed())))

            # SLS
            self.logger.info("SLS")
            local_inc = self.local_search(start_point=start_point)
            
            # decide global inc
            self.logger.info("Race local incumbent against global incumbent")
            self.intensifier.minR = 5 # don't be too aggressive here
            self.intensifier.log_traj = True
            self.incumbent, inc_perf = self.intensifier.intensify(
                    challengers=[local_inc],
                    incumbent=self.incumbent,
                    run_history=self.runhistory,
                    aggregate_func=self.aggregate_func,
                    time_bound=0.01)
            if self.incumbent == local_inc:
                self.logger.info("Changed global incumbent!")

            if self.scenario.shared_model:
                pSMAC.write(run_history=self.runhistory,
                            output_directory=self.scenario.output_dir,
                            num_run=self.num_run)
                
            iteration += 1

            self.logger.debug("Remaining budget: %f (wallclock), %f (ta costs), %f (target runs)" % (
                self.stats.get_remaing_time_budget(),
                self.stats.get_remaining_ta_budget(),
                self.stats.get_remaining_ta_runs()))

            if self.stats.is_budget_exhausted():
                break

            self.stats.print_stats(debug_out=True)

        return self.incumbent
    
    def local_search(self, start_point:Configuration):
        """
        Starts a local search from the given startpoint and quits
        if either the max number of steps is reached or no neighbor
        with an higher improvement was found.

        Parameters:
        ----------

        start_point: Configuration
            The point from where the local search starts

        Returns:
        -------
        incumbent: Configuration
            The best found configuration
        """
        
        self.intensifier.minR = 1 # be aggressive here!
        self.intensifier.log_traj = False
        
        incumbent = start_point
        # Compute the acquisition value of the incumbent
        incumbent_array = convert_configurations_to_array([incumbent])
        acq_val_incumbent = self.acquisition_func(incumbent_array)

        local_search_steps = 0
        neighbors_looked_at = 0
        time_n = []
        while True:

            local_search_steps += 1

            # Get neighborhood of the current incumbent
            # by randomly drawing configurations
            changed_inc = False

            # Get one exchange neighborhood returns an iterator (in contrast of
            # the previously returned list).
            all_neighbors = list(get_one_exchange_neighbourhood(
                incumbent, seed=self.rng.seed()))

            neighbors_array = convert_configurations_to_array(all_neighbors)
            acq_val = self.acquisition_func(neighbors_array)
            
            sorted_neighbors = sorted(zip(all_neighbors, acq_val), key=lambda x: x[1], reverse=True)
            prev_incumbent = incumbent
            
            for neighbor in all_neighbors[:self.MAX_NEIGHBORS]:
                neighbors_looked_at += 1

                self.logger.debug("Intensify")                
                incumbent, inc_perf = self.intensifier.intensify(
                    challengers=[neighbor],
                    incumbent=incumbent,
                    run_history=self.runhistory,
                    aggregate_func=self.aggregate_func,
                    time_bound=0.01)
                
                # first improvement SLS
                if incumbent != prev_incumbent:
                    changed_inc = True
                    break

            if not changed_inc:
                self.logger.info("Local search took %d steps and looked at %d configurations. "% 
                                  (local_search_steps, neighbors_looked_at))
                break

        return incumbent

