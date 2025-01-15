
__global__ void assignPointsToBlocks(double *dataPointsArray, int numAttributes, int numPoints, double *blockMins, double *blockMaxes, int *blockEdges, int numBlocks, int *dataPointBlocks, int *numPointsInBlocks){

    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    // early return here ok, because we'd never have more points than blocks. that would be a disaster.
    if (threadId >= numPoints)
        return;

    // our respective point to compute
    double *thisThreadPoint = &dataPointsArray[threadId * numAttributes];

    // the spot to put our number of which block this data point goes into
    int *dataPointBlock = &dataPointBlocks[threadId];
    *dataPointBlock = -1;

    int currentBlock = 0;
    int nextBlock = 1;
    double *startOfBlockMins, *startOfBlockMaxes, *endOfBlock;

    // now we blast through all the blocks. checking which point this one falls into first
    while (currentBlock < numBlocks){

        // now, we iterate through all the blocks, and the first one our point falls into, we set that block as the value of dataPointBlock, if not we put -1 and the program is broken
        startOfBlockMins = &blockMins[blockEdges[currentBlock]];
        startOfBlockMaxes = &blockMaxes[blockEdges[currentBlock]];
        endOfBlock = &blockMins[blockEdges[nextBlock]];

        bool inThisBlock = true;

        // the x we are at, x0, x1, ...
        int particularAttribute = 0;
        
        // check through all the attributes for this block.
        while(startOfBlockMins < endOfBlock){
           
            // get the amount of x1's that we have in this particular block
            int countOfThisAttribute = (int)*startOfBlockMins;
            
            // increment these two at the same time, since they have the same length and same encoding of number of attributes in them
            startOfBlockMins++;
            startOfBlockMaxes++;
            // now loop that many times, checking if the point is in bounds of any of those intervals
            // we don't actually use i here, because we don't want to check the next attribute of our point on accident. since we may have 2 x2's and such.
            bool inBounds = false;
            for(int i = 0; i < countOfThisAttribute; i++){

                double min = *startOfBlockMins;
                startOfBlockMins++;

                double max = *startOfBlockMaxes;
                startOfBlockMaxes++;

                if(thisThreadPoint[particularAttribute] >= min && thisThreadPoint[particularAttribute] <= max){
                    inBounds = true;
                    break;
                }
            }
            if (!inBounds){
                inThisBlock = false;
                break;
            }
            particularAttribute++;
        }
        // if in this block, we can set dataPointBlock and we're done
        if (inThisBlock){
            *dataPointBlock = currentBlock;
            break;
        }
        // increment the currentBlock and the next block. 
        currentBlock++;
        nextBlock++;   
    }

    // synchronize the threads. now we can move on to summing up the amount of points in each block.
    __syncthreads();

    // now, each block will be represented by one thread, who is going to count up how many points fell into that guy
    if (threadId < numBlocks){
        int count = 0;
        for (int i = 0; i < numPoints; i++){
            if (dataPointBlocks[i] == threadId){
                count++;
            }
        }
        numPointsInBlocks[threadId] = count;
    }

    // sync threads once again, and this time, we are going find the largest block by amount of points, and assign our point to that one there
    __syncthreads();

    // if we found a block for this point, we are going to look if there's a better one, better in this case meaning a block with more points in it.
    if (*dataPointBlock != -1){
        // largest block size is our current block we are assigned to for this point
        int largestBlockSize = numPointsInBlocks[*dataPointBlock];

        // now we blast through all the blocks. checking which point this one falls into first
        while (currentBlock < numBlocks){

            // now, we iterate through all the blocks after the one we chose, and if we find a bigger one we fit into, we go into that one.
            startOfBlockMins = &blockMins[blockEdges[currentBlock]];
            startOfBlockMaxes = &blockMaxes[blockEdges[currentBlock]];
            endOfBlock = &blockMins[blockEdges[nextBlock]];

            bool inThisBlock = true;

            // the x we are at, x0, x1, ...
            int particularAttribute = 0;
            
            // check through all the attributes for this block.
            while(startOfBlockMins < endOfBlock){
            
                // get the amount of x1's that we have in this particular block. this same value should be present in maxes array also.
                int countOfThisAttribute = (int)*startOfBlockMins;
                
                // increment these two at the same time, since they have the same length and same encoding of number of attributes in them
                startOfBlockMins++;
                startOfBlockMaxes++;
                // now loop that many times, checking if the point is in bounds of any of those intervals
                bool inBounds = false;
                for(int i = 0; i < countOfThisAttribute; i++){

                    double min = *startOfBlockMins;
                    startOfBlockMins++;

                    double max = *startOfBlockMaxes;
                    startOfBlockMaxes++;

                    if(thisThreadPoint[particularAttribute] >= min && thisThreadPoint[particularAttribute] <= max){
                        inBounds = true;
                        break;
                    }
                }
                if (!inBounds){
                    inThisBlock = false;
                    break;
                }
                particularAttribute++;
            }
            // if in this block, we can set dataPointBlock and we're done
            if (inThisBlock && numPointsInBlocks[currentBlock] > largestBlockSize){
                *dataPointBlock = currentBlock;
                largestBlockSize = numPointsInBlocks[currentBlock];
            }
            // increment the currentBlock and the next block. 
            currentBlock++;
            nextBlock++;   
        }
    }
    // now we synchronize the threads one last time, and sum up all the amounts once more, since we want to parallelize this task.
    __syncthreads();
    if (threadId < numBlocks){
        int count = 0;
        for (int i = 0; i < numPoints; i++){
            if (dataPointBlocks[i] == threadId){
                count++;
            }
        }
        numPointsInBlocks[threadId] = count;
    }
}