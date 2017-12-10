#!/usr/bin/env python

import numpy as np
import rank_filter2D
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'iterate over different sizes for the average_time_filter function')
    parser.add_argument('inpath', type = str, help = 'Inpath to raw stack')
    parser.add_argument('--startidx', type = int, default = 3, help = 'Start filter size specification')
    parser.add_argument('--stopidx', type = int, default = 9, help = 'Stop filter size specification')
    parser.add_argument('--num', type = int, default = 4, help = 'Number of filter sizes')

    args = parser.parse_args()

    filtszs = np.linspace(args.startidx, args.stopidx, args.num)
    filtszs = filtszs.astype(int)

    for i in filtszs:
        print("Initializing the rank "+ str(i)+" filter with neighborhood size "+str(i)+"x"+str(i))
        separate = ""
        seq = ('/home/zlazri/Documents/spontdata/denoising_blob_folder/Rank_filtered_data/rank', str(i),'_neighbor', str(i),'_stack100.tiff')
        outpath = str(separate.join(seq))
        string = ('os.system("./rank_filter2D.py ', args.inpath, ' ', outpath, ' --sz=', str(i), ' --rank=', str(i), '")')
        parameters = separate.join(string)
        eval(parameters)
