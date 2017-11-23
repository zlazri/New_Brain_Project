#!/bin/bash

# Poisson DWT2 denoising

./stack_as_tiff.py '/home/zlazri/Documents/spontdata/Image_0001_0001.raw' '/home/zlazri/Documents/spontdata/smallstack.tiff'

./denoise_tiff.py '/home/zlazri/Documents/spontdata/smallstack.tiff' '/home/zlazri/Documents/spontdata/denoised_stack.tiff'

./blobs_to_csv.py '/home/zlazri/Documents/spontdata/denoised_stack.tiff' '/home/zlazri/Documents/spontdata/DWT2_stack_data.csv'

# Gaussian filter denoising

./gauss_filter_script.py '/home/zlazri/Documents/spontdata/Image_0001_0001.raw' '/home/zlazri/Documents/spontdata/gauss_filtered_imgs.tiff'

./blobs_to_csv.py '/home/zlazri/Documents/spontdata/gauss_filtered_imgs.tiff' '/home/zlazri/Documents/spontdata/gaussian_stack_data.csv'

# Mean filter denoising

./mean_filter_script.py '/home/zlazri/Documents/spontdata/Image_0001_0001.raw' '/home/zlazri/Documents/spontdata/mean_filtered_imgs.tiff'

./blobs_to_csv.py '/home/zlazri/Documents/spontdata/mean_filtered_imgs.tiff' '/home/zlazri/Documents/spontdata/mean_stack_data.csv'
