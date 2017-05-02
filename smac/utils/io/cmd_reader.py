__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Marius Lindauer"
__email__ = "lindauer@cs.uni-freiburg.de"
__version__ = "0.0.1"

import os
import logging
import numpy
import glob
import typing
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, SUPPRESS


class CMDReader(object):

    """
        use argparse to parse command line options

        Attributes
        ----------
        logger : Logger oject
    """

    def __init__(self):
        """
        Constructor
        """
        self.logger = logging.getLogger(self.__module__ + "." + self.__class__.__name__)
        pass

    def read_cmd(self):
        """
            reads command line options

            Returns
            -------
                args_: parsed arguments; return of parse_args of ArgumentParser
        """

        parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
        req_opts = parser.add_argument_group("Required Options")
        req_opts.add_argument("--scenario_file", required=True,
                              help="scenario file in AClib format")

        req_opts = parser.add_argument_group("Optional Options")
        req_opts.add_argument("--seed", default=12345, type=int,
                              help="random seed")
        req_opts.add_argument("--verbose_level", default=logging.INFO,
                              choices=["INFO", "DEBUG"],
                              help="random seed")
        req_opts.add_argument("--modus", default="SMAC",
                              choices=["SMAC", "ROAR"],
                              help=SUPPRESS)
        
        # list of runhistory dump files and corresponding scenario files
        # format: <scen_file1>@rh_file1,rh_file2,... <scen_file2>@rh_fileN+1,...  
        # scenario corresponding to --warmstart_runhistory; 
        # pcs and feature space has to be identical with --scenario_file
        req_opts.add_argument("--warmstart_runhistory", default=None,
                              nargs="*",
                              help=SUPPRESS)  

        # same format as used for --warmstart_runhistory
        req_opts.add_argument("--warmstart_incumbent", default=None,
                              nargs="*",
                              help=SUPPRESS)# list of trajectory dump files, 
                                            # reads runhistory 
                                            # and uses final incumbent as challenger 
        req_opts.add_argument("--warmstart_mode", default="None",
                              choices = ["None","FULL","WEIGHTED","TRANSFER"],
                              help=SUPPRESS)
        
        req_opts.add_argument("--warmstart_max_read", default=4,
                              type=int,
                              help=SUPPRESS)

        args_, misc = parser.parse_known_args()
        self._check_args(args_)

        # remove leading '-' in option names
        misc = dict((k.lstrip("-"), v.strip("'"))
                    for k, v in zip(misc[::2], misc[1::2]))

        return args_, misc

    def _check_args(self, args_):
        """
            checks command line arguments
            (e.g., whether all given files exist)

            Parameters
            ----------
            args_: parsed arguments
                parsed command line arguments

            Raises
            ------
            ValueError
                in case of missing files or wrong configurations
        """

        if not os.path.isfile(args_.scenario_file):
            raise ValueError("Not found: %s" % (args_.scenario_file))
        
        if args_.warmstart_runhistory:
            args_.warmstart_runhistory = self._convert_warm_args(arg=args_.warmstart_runhistory, 
                                                                 max_to_read=args_.warmstart_max_read)
        if args_.warmstart_incumbent:
            args_.warmstart_incumbent = self._convert_warm_args(arg=args_.warmstart_incumbent, 
                                                                max_to_read=args_.warmstart_max_read)
             
    def _convert_warm_args(self, arg:typing.List[str], max_to_read:int=4):
        conv_dict = {}
        for entry in arg:
            scen_fn, files_ = entry.split("@")
            if conv_dict.get(scen_fn):
                self.logger.warn("Redefined data for scenario %s" %(scen_fn))
            conv_dict[scen_fn] = []
            for file_ in files_.split(","):
                conv_dict[scen_fn].extend(glob.glob(file_))
            if len(conv_dict[scen_fn]) > max_to_read:
                conv_dict[scen_fn] = random.sample(conv_dict[scen_fn],max_to_read)
        return conv_dict