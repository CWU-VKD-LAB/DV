#!/bin/bash
nvcc -ptx -o SimplificationsKernels.ptx SimplificationsKernels.cu
nvcc -ptx -o RemoveUselessBlocksCuda.ptx RemoveUselessBlocksCuda.cu
nvcc -ptx -o MergerHyperKernels.ptx MergerHyperKernels.cu