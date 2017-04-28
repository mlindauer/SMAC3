import sys
import os
import glob
import re
import json
import tabulate
del(tabulate.LATEX_ESCAPE_RULES[u'$'])
del(tabulate.LATEX_ESCAPE_RULES[u'\\'])
del(tabulate.LATEX_ESCAPE_RULES[u'{'])
del(tabulate.LATEX_ESCAPE_RULES[u'}'])
del(tabulate.LATEX_ESCAPE_RULES[u'^'])


import numpy as np
import matplotlib
matplotlib.use('Agg')
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)

import matplotlib.pyplot as plt

regex_warm = re.compile("INFO:ChallengerWarmstart:(?P<idx>\d+) : .* \((?P<perc>.*) perc\)")#
regex_change = re.compile("INFO:intensifier:Challenger \(.*\) is better than incumbent (.*) on (?P<runs>\d+) runs\.")

scen_dict = {"CSSC14_industrial_CSSC-BMC08-300s-2day_conti_SparrowToRiss": "BMC",
             "CSSC14_industrial_CSSC-CircuitFuzz-300s-2day_conti_SparrowToRiss": "CircuitFuzz",
             "CSSC14_industrial_CSSC-IBM-300s-2day_conti_SparrowToRiss": "IBM",
             "CSSC14_random_CSSC-3cnf-v350-300s-2day_conti_minisat-HACK-999ED-CSSC" : "3cnf",
             "CSSC14_random_CSSC-K3-300s-2day_conti_minisat-HACK-999ED-CSSC": "K3",
             "CSSC14_random_CSSC-unsat-unif-k5-300s-2day_conti_minisat-HACK-999ED-CSSC": "unsat-k5"
    }

tracks = [["BMC","CircuitFuzz","IBM"],["3cnf","K3","unsat-k5"]]
tracks = [["CSSC14_industrial_CSSC-BMC08-300s-2day_conti_SparrowToRiss", "CSSC14_industrial_CSSC-CircuitFuzz-300s-2day_conti_SparrowToRiss",
           "CSSC14_industrial_CSSC-IBM-300s-2day_conti_SparrowToRiss"],
          ["CSSC14_random_CSSC-3cnf-v350-300s-2day_conti_minisat-HACK-999ED-CSSC",
           "CSSC14_random_CSSC-K3-300s-2day_conti_minisat-HACK-999ED-CSSC",
           "CSSC14_random_CSSC-unsat-unif-k5-300s-2day_conti_minisat-HACK-999ED-CSSC"]]


scen_data = {}
for scenario in os.listdir():
    if not os.path.isdir(scenario):
        continue
    
    print(scenario)

    log_fns = glob.glob("%s/SMAC3_warm_--warmstart_incumben/run*/log-*.txt" %(scenario))

    for fn in log_fns:
        if "log-val" in fn:
            continue
        print(fn)
        scen_order = []
        inc_from = []   
        inc_sources = ["default"] 
        with open(fn) as fp:
            next(fp)
            for line in fp:
                if line.startswith("INFO:scenario:Reading scenario file"):
                    scen = line.split("/")[-2]
                    if scen not in scen_order:
                        scen_order.extend([scen]*20)
                match = regex_warm.match(line)
                if match:
                    idx = int(match.group("idx")) 
                    try:
                        scen = scen_order.pop(idx)
                    except:
                        print("WARNING: Could not pop %d idx" %(idx))
                    inc_from.append(scen)
                match = regex_change.match(line)
                if match:
                    runs = int(match.group("runs"))
                    runs -= 3 # first change needs 3 runs
                    try:
                        inc_sources.append(inc_from[runs])
                    except IndexError:
                        pass
        
        print(inc_sources)
        
        run_id = int(fn.split("/")[-2].split("-")[1])
        # read traj file to get time stamps
        traj_fn = glob.glob("%s/SMAC3_warm_--warmstart_incumben/run-%d/smac3-output*/traj_aclib2.json" %(scenario,run_id,))[0]
        time_stemps = []
        with open(traj_fn) as fp:
            for idx_, line in enumerate(fp):
                if idx_ == len(inc_sources):
                    break
                traj_entry = json.loads(line)
                time_stemps.append(traj_entry["wallclock_time"])
        
        # get validation file
        val_fn = glob.glob("%s/SMAC3_warm_--warmstart_incumben/run-%d/validate-time-test/validationResults-traj_old-walltimeworker.csv" %(scenario,run_id,))[0]
        traj_perfs = []
        max_impr = {}
        with open(val_fn) as fp:
            next(fp)
            def_line = next(fp)
            _,_,def_perf,_,_,_ = def_line.split(",")
            def_perf = float(def_perf)
            for line in fp:
                t,_,perf,_,_,_ = line.split(",")
                perf = float(perf)
                t = float(t)
                if time_stemps and t >= time_stemps[0]:
                    while time_stemps and t >= time_stemps[0]:
                        time_stemps.pop(0)
                        source = inc_sources.pop(0)
                    max_impr[source] = max(max_impr.get(source,0), def_perf - perf)
        try:
            max_impr = dict([[s, p/(def_perf-perf) * 100] for s,p in max_impr.items()])
        except ZeroDivisionError:
            pass
        
        for s,p in max_impr.items():
            if p < -100:
                max_impr[s] = 0 # clean broken validation
        
        if sum(max_impr.values()) == 0: # no data - broken run
            continue
        scen_data[scenario] = scen_data.get(scenario,[])
        scen_data[scenario].append(max_impr)
        
print(scen_data["CSSC14_industrial_CSSC-BMC08-300s-2day_conti_SparrowToRiss"])

for track in tracks:
    table = []
    for scen_i in track:
        row = [scen_dict.get(scen_i,scen_i)]
        for scen_j in track:
            m = np.mean([r.get(scen_j,0) for r in scen_data[scen_i]])
            row.append("$%.1f$" %(m))
        table.append(row)
    print(tabulate.tabulate(tabular_data=table, headers=[""]+[scen_dict.get(scen,scen) for scen in track], tablefmt="latex_booktabs"))
        
