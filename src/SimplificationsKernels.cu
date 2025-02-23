extern "C"


// Global memory: Slow, can be accessed by all threads regardless of block. Slow!!
// Shared memory: Shared by all threads in the same block, much faster than global memory.
__global__ void removeUselessHelper(float* mins, float* maxes, int* intervalCounts,
    int minMaxLen, int* blockEdges, int numBlocks, int* blockClasses,
    char* attrRemoveFlags, int fieldLen, float* dataset, int numPoints,
    int* classBorder, int numClasses) {

    int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    if(threadID >= numBlocks) return;

    // Get block-specific data
    float* blockMins = &mins[blockEdges[threadID]];
    float* blockMaxes = &maxes[blockEdges[threadID]];
    int* blockIntervalCounts = &intervalCounts[threadID * fieldLen];
    int classNum = blockClasses[threadID];

    // Class boundaries
    int startClass = classBorder[classNum] * fieldLen;
    int endClass = classBorder[classNum+1] * fieldLen;

    // Try removing each attribute
    for(int removed = 0; removed < fieldLen; removed++) {
        // Calculate offset to the start of this attribute's intervals
        int checkOffset = 0;
        for(int i = 0; i < removed; i++) {
            checkOffset += blockIntervalCounts[i];
        }

        // Skip if this attribute is already marked as removed (checking first interval)
        if(blockMins[checkOffset] == 0.0 && blockMaxes[checkOffset] == 1.0) {
            continue;
        }

        bool someOneInBounds = false;
        // Check all peoints from other classes
        for(int j = 0; j < numPoints * fieldLen; j += fieldLen) {
            if(j < endClass && j >= startClass) continue;

            bool pointInside = true;
            int totalOffset = 0;

            // Check each attribute
            for(int attr = 0; attr < fieldLen; attr++) {
                if(attr == removed) {
                    totalOffset += blockIntervalCounts[attr];
                    continue;
                }

                float attrValue = dataset[j + attr];
                bool inAnInterval = false;

                for(int intv = 0; intv < blockIntervalCounts[attr]; intv++) {
                    float min = blockMins[totalOffset + intv];
                    float max = blockMaxes[totalOffset + intv];

                    if(attrValue <= max && attrValue >= min) {
                        inAnInterval = true;
                        break;
                    }
                }

                if(!inAnInterval) {
                    pointInside = false;
                    break;
                }

                totalOffset += blockIntervalCounts[attr];
            }

            if(pointInside) {
                someOneInBounds = true;
                break;
            }
        }

        // If no points from other classes fall in, we can remove this attribute
        if(!someOneInBounds) {
            int removeOffset = 0;
            for(int i = 0; i < removed; i++) {
                removeOffset += blockIntervalCounts[i];
            }

            // Reset intervals for removed attribute to [0,1]
            for(int i = 0; i < blockIntervalCounts[removed]; i++) {
                blockMins[removeOffset + i] = 0.0;
                blockMaxes[removeOffset + i] = 1.0;
            }

            // Mark attribute as removed
            attrRemoveFlags[fieldLen * threadID + removed] = 1;
        }
    }
}

// Local memory: Private to each thread. Used for storing variables that don't fit in registers
__global__ void removeUselessHelperDouble(double* mins, double* maxes, int* intervalCounts,
    int minMaxLen, int* blockEdges, int numBlocks, int* blockClasses,
    char* attrRemoveFlags, int fieldLen, double* dataset, int numPoints,
    int* classBorder, int numClasses) {

    int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    if(threadID >= numBlocks) return;

    // Get block-specific data
    double* blockMins = &mins[blockEdges[threadID]];
    double* blockMaxes = &maxes[blockEdges[threadID]];
    int* blockIntervalCounts = &intervalCounts[threadID * fieldLen];
    int classNum = blockClasses[threadID];

    // Class boundaries
    int startClass = classBorder[classNum] * fieldLen;
    int endClass = classBorder[classNum+1] * fieldLen;

    // Try removing each attribute
    for(int removed = 0; removed < fieldLen; removed++) {
        // Calculate offset to the start of this attribute's intervals
        int checkOffset = 0;
        for(int i = 0; i < removed; i++) {
            checkOffset += blockIntervalCounts[i];
        }

        // Skip if this attribute is already marked as removed (checking first interval)
        if(blockMins[checkOffset] == 0.0 && blockMaxes[checkOffset] == 1.0) {
            continue;
        }

        bool someOneInBounds = false;
        // Check all peeeeeoints from other classes
        for(int j = 0; j < numPoints * fieldLen; j += fieldLen) {
            if(j < endClass && j >= startClass) continue;

            bool pointInside = true;
            int totalOffset = 0;

            // Check each attribute
            for(int attr = 0; attr < fieldLen; attr++) {
                if(attr == removed) {
                    totalOffset += blockIntervalCounts[attr];
                    continue;
                }

                double attrValue = dataset[j + attr];
                bool inAnInterval = false;

                for(int intv = 0; intv < blockIntervalCounts[attr]; intv++) {
                    double min = blockMins[totalOffset + intv];
                    double max = blockMaxes[totalOffset + intv];

                    if(attrValue <= max && attrValue >= min) {
                        inAnInterval = true;
                        break;
                    }
                }

                if(!inAnInterval) {
                    pointInside = false;
                    break;
                }

                totalOffset += blockIntervalCounts[attr];
            }

            if(pointInside) {
                someOneInBounds = true;
                break;
            }
        }

        // If no points from other classes fall in, we can remove this attribute
        if(!someOneInBounds) {
            int removeOffset = 0;
            for(int i = 0; i < removed; i++) {
                removeOffset += blockIntervalCounts[i];
            }

            // Reset intervals for removed attribute to [0,1]
            for(int i = 0; i < blockIntervalCounts[removed]; i++) {
                blockMins[removeOffset + i] = 0.0;
                blockMaxes[removeOffset + i] = 1.0;
            }

            // Mark attribute as removed
            attrRemoveFlags[fieldLen * threadID + removed] = 1;
        }
    }
}