#!/bin/bash
#nvcc -ptx -o SimplificationsKernels.ptx SimplificationsKernels.cu
#nvcc -ptx -o RemoveUselessBlocksCuda.ptx RemoveUselessBlocksCuda.cu
#nvcc -ptx -o MergerHyperKernels.ptx MergerHyperKernels.cu -lcudadevrt
nvcc -ptx -arch=sm_75 -rdc=true -o MergerHyperKernels.ptx MergerHyperKernels.cu
