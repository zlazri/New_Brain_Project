#!/bin/bash

./stack_as_tiff.py '/home/zlazri/Documents/spontdata/Image_0001_0001.raw' '/home/zlazri/Documents/spontdata/smallstack.tiff'

./denoise_tiff.py '/home/zlazri/Documents/spontdata/smallstack.tiff' '/home/zlazri/Documents/spontdata/denoised_stack.tiff'

./blobs_to_csv.py '/home/zlazri/Documents/spontdata/denoised_stack.tiff' '/home/zlazri/Documents/spontdata/stack_data.csv'
