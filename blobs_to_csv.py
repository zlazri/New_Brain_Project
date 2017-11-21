#!/usr/bin/env python

import numpy as np
import argparse
import matplotlib as plt
from matplotlib import pyplot as plt
from libtiff import TIFF
import skimage
from skimage.feature import blob_log
from skimage.feature import blob_dog
from skimage import exposure
import csv
from numpy import sqrt

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'Perform blob detection and write results to csv')
    parser.add_argument('inpath', type = str, help = 'Inpath to tiff stack')
    parser.add_argument('outpath', type = str, help = 'Outpath for blob location results to a csv file')
    parser.add_argument('--method', type=str, default = 'bloblog', help = 'Specify blob detection method-- blobdog, bloblog, or blobdoh')
    parser.add_argument('--method_opts', type=tuple, default = (10, 5, 10, .085), help= 'Specify parameters for the blob detection method. tuple is of format: (max_sigma, min_sigma, num_sigma, threshold). See scikit for meaning of these parameters') 

    args = parser.parse_args()

    if args.method == 'blobdog':
        blob_method = blob_dog
    elif args.method == 'bloblog':
        blob_method = blob_log
    else:
        blob_method = blob_doh

    tiff = TIFF.open(args.inpath, mode='r')
    with open(args.outpath, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        for image in tiff.iter_images():
            if args.method == 'bloblog':
                blobdata = blob_log(image, max_sigma = args.method_opts[0], min_sigma = args.method_opts[1], num_sigma = args.method_opts[2], threshold = args.method_opts[3])
                blobdata[:, 2] = blobdata[:, 2] * sqrt(2)
            elif args.method == 'blobdog':
                blobdata = blob_dog(image, max_sigma = args.method_opts[0], min_sigma = args.method_opts[1],  threshold = args.method_opts[3])
                blobdata[:, 2] = blobdata[:, 2] * sqrt(2)
            else:
                blobdata = blob_doh(image, max_sigma = args.method_opts[0], min_sigma = arg.method_opts[1], threshold= args.method_opts[3])

            spamwriter.writerow([blobdata])


#----------------Sample blob image------------------------
            
#            fig, axes = plt.subplots(1,1)
#
#            size1 = np.shape(blobdata)[0]
#
#            axes.imshow(image)
#            for i in range(size1):
#                y, x, r = blobdata[i,:]
#                c = plt.Circle((x, y), r, color="red", linewidth=2, fill=False)
#                axes.add_patch(c)
#
#            plt.show()
#            assert(1==0)
