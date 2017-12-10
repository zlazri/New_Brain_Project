#!/bin/bash

# Poisson DWT2 (variance stablizing) denoising

./stack_as_tiff.py '/home/zlazri/Documents/spontdata/Image_0001_0001.raw' '/home/zlazri/Documents/spontdata/500stack.tiff'

./denoise_tiff.py '/home/zlazri/Documents/spontdata/500stack.tiff' '/home/zlazri/Documents/spontdata/denoised_stack.tiff'

./blobs_to_csv.py '/home/zlazri/Documents/spontdata/denoised_stack.tiff' '/home/zlazri/Documents/spontdata/DWT2_stack_data.csv'


# Gaussian filter denoising (2D)

./gauss_filter_script.py '/home/zlazri/Documents/spontdata/Image_0001_0001.raw' '/home/zlazri/Documents/spontdata/gauss_filtered_imgs.tiff'

./blobs_to_csv.py '/home/zlazri/Documents/spontdata/gauss_filtered_imgs.tiff' '/home/zlazri/Documents/spontdata/gaussian_stack_data.csv'


# Mean filter denoising (2D)

./mean_filter_script.py '/home/zlazri/Documents/spontdata/Image_0001_0001.raw' '/home/zlazri/Documents/spontdata/mean_filtered_imgs.tiff'

./blobs_to_csv.py '/home/zlazri/Documents/spontdata/mean_filtered_imgs.tiff' '/home/zlazri/Documents/spontdata/mean_stack_data.csv'


# Rank filter denoising (2D)

./rank_filter_ranges.py '/home/zlazri/Documents/spontdata/chan2_Image_0001_0001.raw' --startidx=3 --stopidx=9 --num=4


# 4Dtensor denoising (2D)


# Ground truth denoising (2D)

./average_time_filter.py '/home/zlazri/Documents/spontdata/chan2_Image_0001_0001.raw' '/home/zlazri/Documents/spontdata/denoising_blob_folder/Ground_truth_data/averaged_n100.tiff'


# FFT reg average_time_data (2D)

./time_average_ranges.py '/home/zlazri/Documents/spontdata/chan2_Image_0001_0001.raw' '/home/zlazri/Documents/spontdata/denoising_blob_folder/FFTreg_average_time_data/FFTreg_stack_n' --startidx=3 --stopidx=9 --num=4
