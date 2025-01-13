extern "C"


// Global memory: Slow, can be accessed by all threads regardless of block. Slow!!

// Shared memory: Shared by all threads in the same block, much faster than global memory.

// Local memory: Private to each thread. Used for storing variables that don't fit in registers


// This will parallelize the removeUselessAttributes
__global__ void removeUselessHelper(){

}

__device__ boolean insideHBIntervals(){


    return inside;
}
