from smac.runhistory.runhistory import RunHistory
from smac.scenario.scenario import Scenario
from smac.configspace import ConfigurationSpace
from smac.smbo.objective import average_cost

import typing

def merge_foreign_data_from_file(scenario: Scenario, 
                                 runhistory: RunHistory,
                                 in_scenario_fn_list: typing.List[str], 
                                 in_runhistory_fn_list: typing.List[str],
                                 cs: ConfigurationSpace,
                                 aggregate_func: typing.Callable = average_cost,
                                 update_train:bool=False):
    '''
        extend <scenario> and <runhistory> with runhistory data from another <in_scenario> 
        assuming the same pcs, feature space, but different instances

        Arguments
        ---------
        scenario: Scenario
            original scenario -- feature dictionary will be extended
        runhistory: RunHistory
            original runhistory -- will be extended by further data points
        in_scenario_fn_list: typing.List[str]
            input scenario file names
        in_runhistory_fn_list: typing.List[str]
            list filenames of runhistory dumps
        cs: ConfigurationSpace
            parameter configuration space to read runhistory from file
        aggregate_func: typing.Callable
            function to aggregate performance of a configuratoion across instances
        update_train: bool
            adds instances from read scenarios to training instances in returned scenario

        Returns
        -------
            scenario, runhistory
    '''

    if not in_scenario_fn_list:
        raise ValueError("To read warmstart data from previous runhistories, the corresponding scenarios are required. Use option --warmstart_scenario") 
    scens = [Scenario(scenario=scen_fn, cmd_args={"output_dir":""}) for scen_fn in in_scenario_fn_list]
    rhs = []
    for rh_fn in in_runhistory_fn_list:
        rh = RunHistory(aggregate_func)
        rh.load_json(rh_fn, cs)
        rhs.append(rh)

    return merge_foreign_data(scenario, runhistory, in_scenario_list=scens, in_runhistory_list=rhs,
                              update_train=update_train)


def merge_foreign_data(scenario: Scenario, 
                       runhistory: RunHistory,
                       in_scenario_list: typing.List[Scenario], 
                       in_runhistory_list: typing.List[RunHistory],
                       update_train:bool=False):
    '''
        extend <scenario> and <runhistory> with runhistory data from another <in_scenario> 
        assuming the same pcs, feature space, but different instances

        Arguments
        ---------
        scenario: Scenario
            original scenario -- feature dictionary will be extended
        runhistory: RunHistory
            original runhistory -- will be extended by further data points
        in_scenario_list: typing.List[Scenario]
            input scenario 
        in_runhistory_list: typing.List[RunHistory]
            list of runhistories wrt <in_scenario>
        update_train: bool
            adds instances from read scenarios to training instances in returned scenario

        Returns
        -------
            scenario, runhistory
    '''

    # add further instance features
    for in_scenario in in_scenario_list:
        if scenario.n_features != in_scenario.n_features:
            raise ValueError("Feature Space has to be the same for both scenarios (%d vs %d)." % (
                scenario.n_features, in_scenario.n_features))

        if scenario.cs != in_scenario.cs:
            raise ValueError("PCS of both scenarios have to be identical.")

        if scenario.cutoff != in_scenario.cutoff:
            raise ValueError("Cutoffs of both scenarios have to be identical.")

        scenario.feature_dict.update(in_scenario.feature_dict)
        
        if update_train:
            scenario.train_insts = list(set(scenario.train_insts).union(in_scenario.train_insts))

    # extend runhistory
    for rh in in_runhistory_list:
        runhistory.update(rh, external_data=True)

    for date in runhistory.data:
        if scenario.feature_dict.get(date.instance_id) is None:
            raise ValueError(
                "Instance feature for \"%s\" was not found in scenario data." % (date.instance_id))

    runhistory.compute_all_costs(instances=scenario.train_insts)
    

    return scenario, runhistory
