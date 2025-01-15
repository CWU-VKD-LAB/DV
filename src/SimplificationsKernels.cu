extern "C"


// Global memory: Slow, can be accessed by all threads regardless of block. Slow!!

// Shared memory: Shared by all threads in the same block, much faster than global memory.

// Local memory: Private to each thread. Used for storing variables that don't fit in registers


__global__ void removeUselessHelper(double* mins, double* maxes, int* intervalCounts, int minMaxLen, int* blockEdges, int numBlocks, int* blockClasses, char* attrRemoveFlags, int fieldLen, double* dataset, int numPoints, int* classBorder, int numClasses){
    // Get which thread we are on
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    // Kill unneeded threads.
    if(threadID >= numBlocks){
        return;
    }

    // The threadID will tell us which block we need to process.
    int classNum = blockClasses[threadID];

    // We will try to remove each attribute from the block.
    for(int removed = 0; removed < fieldLen; removed++){

        // Find the first encoding of mins and maxes
        double* startOfMins = &mins[blockEdges[threadID]];
        double* startOfMaxes = &maxes[blockEdges[threadID]];
        double* end = &mins[blockEdges[threadID + 1]];

        // Find the index the class point array stops and ends.
        int startClass = classBorder[classNum] * fieldLen;
        int endClass = classBorder[classNum+1] * fieldLen;

        // some point in bounds initialized to falase;
        bool someOneInBounds = false;

         // Go through all points that aren't of same class
        for(int j = 0; j < numPoints * fieldLen; j += fieldLen){

            // We want to skip the range given by classBorder[classNum]
            if(j < endClass && j >= startClass){
                continue;
            }

            bool pointInside = true;
            int currAttr = 0;

            // Go through all the attributes.
            while(startOfMins < end){
                // I need to instead look for an attribute that the point is not inside any interval for
                // Get number of intervals and push past the encoded num.
                int numIntervals = (int) *startOfMins;

                startOfMins++;
                startOfMaxes++;

                // If at the attribute we are going to remove, skip past it
                if(currAttr == removed){
                    for(int k = 0; k < numIntervals && startOfMins < end; k++){
                        startOfMins++;
                        startOfMaxes++;
                    }
                    currAttr++;
                    // Just skip to next iteration, it is simpler.
                    continue;
                }

                // Point starts out being in no intervals?
                bool inAnInterval = false;

                 // Loop through all the intervals for this attribute.
                for(int i = 0; i < numIntervals; i++){
                    double min = *startOfMins;
                    double max = *startOfMaxes;
                    startOfMins++;
                    startOfMaxes++;

                    double attrValue = dataset[j+currAttr];

                    // Check if point attribute in an interval of the attribute.
                    if(attrValue <= max && attrValue >= min){
                        inAnInterval = true;
                        break;
                    }
                }

                if(!inAnInterval){
                    // Reset the mins/maxes back to the start.
                    startOfMins = &mins[blockEdges[threadID]];
                    startOfMaxes = &maxes[blockEdges[threadID]];
                    pointInside = false;
                    break;
                }

                currAttr++;
            }

            // If the point is inside, stop going through points and cancel the remove.
            if(pointInside){
                someOneInBounds = true;
                break;
            }
        }

        // If no points from other classes fell in, we can removed the current attribute.
        if(!someOneInBounds){
            int currAttr = 0;
            // Get the very start of the block
            startOfMins = &mins[blockEdges[threadID]];
            startOfMaxes = &maxes[blockEdges[threadID]];

            while (startOfMins < end) {
                int n = (int) *startOfMins;  // Get the number of intervals for the current attribute
                startOfMins++;  // Move past the interval count
                startOfMaxes++; // Move past the interval count for max values

                if (currAttr == removed) {
                    // If we are at the attribute to remove, set all its intervals to [0, 1]
                    for (int i = 0; i < n; i++) {
                        *startOfMins = 0.0;  // Set the min value to 0
                        *startOfMaxes = 1.0; // Set the max value to 1
                        startOfMins++;  // Move to the next interval
                        startOfMaxes++; // Move to the next interval
                    }
                } else {
                    // Skip over the intervals for attributes that are not removed
                    for (int i = 0; i < n; i++) {
                        startOfMins++;  // Move past the min value for this interval
                        startOfMaxes++; // Move past the max value for this interval
                    }
                }
                currAttr++;
            }

            attrRemoveFlags[fieldLen * threadID + removed] = 1;
        }

    }
    return;
}