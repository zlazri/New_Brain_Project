#!/usr/bin/env python

import numpy as np
import average_time_filter
import argparse
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'iterate over different sizes for the average_time_filter function')
    parser.add_argument('inpath', type = str, help = 'Inpath to raw stack')
    parser.add_argument('baseoutpath', type = str, help = 'Base outpath name. Multiply outpaths will be created, with names of the form baseoutpath_num')
    parser.add_argument('--startidx', type = int, default = 3, help = 'Start filter size specification')
    parser.add_argument('--stopidx', type = int, default = 9, help = 'Stop filter size specification')
    parser.add_argument('--num', type = int, default = 4, help = 'Number of filter sizes')

    args = parser.parse_args()

    filtszs = np.linspace(args.startidx, args.stopidx, args.num)
    filtszs = filtszs.astype(int)

    for i in filtszs:
        separate = ""
        seq = (args.baseoutpath, str(i),'.tiff')
        outpath = str(separate.join(seq))
        string = ('os.system("./average_time_filter.py ', args.inpath, ' ', outpath, ' --size=', str(i), '")')
        parameters = separate.join(string)
        eval(parameters)
