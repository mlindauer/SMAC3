import sys
import os
import glob
import re

import numpy as np
import matplotlib.pyplot as plt


for scenario in os.listdir():
    if not os.path.isdir(scenario):
        continue
    
    print(scenario)

    log_fns = glob.glob("%s/SMAC3_warm_--warmstart_mode_WEI/run*/log-*.txt" %(scenario))
    
    float_regex = '[+-]?\d+(?:\.\d+)?(?:[eE][+-]\d+)?'
    regex_str = "INFO:WarmstartedRandomForestWithInstances:Model weights: \[[ ]*(?P<w1>{0})[ ]*(?P<w2>{0})[ ]*(?P<w3>{0})[ ]*\] \+ intercept: (?P<intercept>{0})".format(float_regex)
    reg = re.compile(regex_str)
    
    all_data = []
    for log_fn in log_fns:
        data = []
        with open(log_fn) as fp:
            for line in fp:
                match = reg.match(line)
                if match:
                    w1 = float(match.group("w1"))
                    w2 = float(match.group("w2"))
                    w3 = float(match.group("w3"))
                    d = [w1,w2,w3]
                    data.append(d)
        if data:
            all_data.append(data)
        
    all_data = all_data
    
    colors = ["r","b","g"]
    
    print("Runs: %d" %(len(all_data)))
    for run in all_data:
        run = np.array(run)
        x = np.linspace(0,1,run.shape[0])
        for w in range(run.shape[1]):
            y = run[:,w]
            plt.plot(x,y,c=colors[w],alpha=0.4)

    plt.xlabel("Fraction of configuration budget")
    plt.ylabel("Weight")            
    plt.title(scenario)
    plt.savefig("%s_weights.pdf" %(scenario))
    plt.close()