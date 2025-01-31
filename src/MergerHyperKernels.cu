extern "C"

#define max(a, b) (a > b)? a: b
#define min(a, b) (a > b)? b: a

// our basic procedure is this. start with block 0 as a seed. try and merge all the other guys to block 0. if anyone does, then we can mark 0 for deletion.
// iterate to the next one, merging the blocks to that one and so on.
// we have only passed in countercases, not cases of our own class, all the blocks we are testing are also already the same class.
// may need some changing around, when we test with a dataset over 2000 ish attributes we can't use shared memory to store the seedblock attributes, so we just will use global if such a thing happens.
__global__ void MergerHelper1(float *hyperBlockMins, float *hyperBlockMaxes, float *combinedMins, float *combinedMaxes, int *deleteFlags, int numAttributes, float *points, int numPoints, int numBlocks, int* seedQueue){
    // shared memory is going to be 2 * numAttributes * sizeof(float) long. this lets us load the whole seed block into memory
    extern __shared__ float seedBlockAttributes[];
    __syncthreads();

    // stored in the same array, but using two pointers we can make this easy.
    float *seedBlockMins = &seedBlockAttributes[0];
    float *seedBlockMaxes = &seedBlockAttributes[numAttributes];

    // get our thread id
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    __syncthreads();

    float *thisBlockCombinedMins = &combinedMins[threadID * numAttributes];
    float *thisBlockCombinedMaxes = &combinedMaxes[threadID * numAttributes];
    __syncthreads();

    int positionInSeedQueue = threadID;
    __syncthreads();

    bool mergable = true;
    for(int deadSeedNum = 0; deadSeedNum < numBlocks; deadSeedNum++){
        __syncthreads();
        // This is the thing with accessing the right block from the queue.
        int seedBlock = seedQueue[deadSeedNum];
        __syncthreads();

        // copy the seed blocks attributes into shared memory, so that we can load stuff much faster.
        // block Dim is how many threads in a block. since each block has it's own shared memory, this is our offset for copying stuff over. this is needed because
        // we could have more than 1024 attributes very easily.
        for (int index = threadIdx.x; index < numAttributes; index += blockDim.x){
            seedBlockMins[index] = hyperBlockMins[seedBlock * numAttributes + index];
            seedBlockMaxes[index] = hyperBlockMaxes[seedBlock * numAttributes + index];
        }
        // sync when we're done copying over the seedblock values.
        __syncthreads();

        // we don't have to test blocks before the seedblock, since they've already been tested
        // also we aren't going to have a block if the threadID is too big.
        // make the combined mins and maxes, and then check against all our data.
        if (threadID != seedBlock && threadID < numBlocks && deleteFlags[threadID] >= 0 ){

            // first we build our combined list.
            for (int i = 0; i < numAttributes; i++){
                thisBlockCombinedMaxes[i] = max(seedBlockMaxes[i], hyperBlockMaxes[threadID * numAttributes + i]);
                thisBlockCombinedMins[i] = min(seedBlockMins[i], hyperBlockMins[threadID * numAttributes + i]);
            }

            // now we check all our data for a point falling into our new bounds.
            char allPassed = 1;
            for (int point = 0; point < numPoints; point++){

                char someAttributeOutside = 0;
                for(int att = 0; att < numAttributes; att++){
                    if (points[point * numAttributes + att] > thisBlockCombinedMaxes[att] || points[point * numAttributes + att] < thisBlockCombinedMins[att]){
                        someAttributeOutside = 1;
                        break;
                    }
                }
                // if there's NOT some attribute outside, this point has fallen in, and we can't do the merge.
                if (!someAttributeOutside){
                    allPassed = 0;
                    break;
                }
            }
            // if we did pass all the points, that means we can merge, and we can set the updated mins and maxes for this point to be the combined attributes instead.
            // then we simply flag that seedBlock is trash.
            if (allPassed && deleteFlags[threadID] >= 0){
                // copy the combined mins and maxes into the original array
                for (int i = 0; i < numAttributes; i++){
                    hyperBlockMins[threadID * numAttributes + i] = thisBlockCombinedMins[i];
                    hyperBlockMaxes[threadID * numAttributes + i] = thisBlockCombinedMaxes[i];
                }
                // set the flag to -1. atomic because many threads will try this.

                atomicMin(&deleteFlags[seedBlock], -1);
                atomicMax(&deleteFlags[threadID], 1);
            }
        } // checking one seedblock loop
        __syncthreads();


        if(threadID == 0 && deleteFlags[seedBlock] != -1){
            deleteFlags[seedBlock] = -9;
        }

        __syncthreads();

        // Redo the order of the non-existent queue seedQueue
        if(threadID < numBlocks){
            int cnt = 0;

            if(deleteFlags[threadID] == 1){
                // Count how many 1's are to the left of me, put self
                for(int i = positionInSeedQueue - 1; i >= 0; i--){
                    if(deleteFlags[seedQueue[i]] == 1){
                        cnt++;
                    }
                }
                positionInSeedQueue = numBlocks - 1 - cnt;
            }
            else if(deleteFlags[threadID] == 0){
               // count all non-1's to the left, then put self in deadSeedNum + count
               for(int i = positionInSeedQueue - 1; i >= 0; i--){
                    if(deleteFlags[seedQueue[i]] != 1){
                        cnt++;
                    }
               }
               positionInSeedQueue = cnt;
            }
        }

        __syncthreads();
        if(threadID < numBlocks){
            seedQueue[positionInSeedQueue] = threadID;
        }

        __syncthreads();
        //RESET THE DELETE FLAGS ALL 1 -> 0, LEAVE -1 and -9 ALONE
        if(threadID < numBlocks && deleteFlags[threadID] == 1){
            deleteFlags[threadID] = 0;
        }
        __syncthreads();
    }
}




// our basic procedure is this. start with block 0 as a seed. try and merge all the other guys to block 0. if anyone does, then we can mark 0 for deletion.
// iterate to the next one, merging the blocks to that one and so on.
// we have only passed in countercases, not cases of our own class, all the blocks we are testing are also already the same class.
/* may need some changing around, when we test with a dataset over 2000 ish attributes we can't use shared memory to store the seedblock attributes, so we just will use global if such a thing happens.
__global__ void MergerHelper1(float *hyperBlockMins, float *hyperBlockMaxes, float *combinedMins, float *combinedMaxes, int *deleteFlags, int numAttributes, float *points, int numPoints, int numBlocks){

    // shared memory is going to be 2 * numAttributes * sizeof(float) long. this lets us load the whole seed block into memory
    extern __shared__ float seedBlock[];

    // stored in the same array, but using two pointers we can make this easy.
    float *seedBlockMins = &seedBlock[0];
    float *seedBlockMaxes = &seedBlock[numAttributes];

    // get our thread id
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    float *thisBlockCombinedMins = &combinedMins[threadID * numAttributes];
    float *thisBlockCombinedMaxes = &combinedMaxes[threadID * numAttributes];
    //printf("Thread %d   :   Blocks : %d     Num Points:  %d\n", threadID, numBlocks, numPoints);
    for(int seedBlockNum = 0; seedBlockNum < numBlocks; seedBlockNum++){

        // copy the seed blocks attributes into shared memory, so that we can load stuff much faster.
        // block Dim is how many threads in a block. since each block has it's own shared memory, this is our offset for copying stuff over. this is needed because
        // we could have more than 1024 attributes very easily.
        for (int index = threadIdx.x; index < numAttributes; index += blockDim.x){
            seedBlockMins[index] = hyperBlockMins[seedBlockNum * numAttributes + index];
            seedBlockMaxes[index] = hyperBlockMaxes[seedBlockNum * numAttributes + index];
        }
        // sync when we're done copying over the seedblock values.
        __syncthreads();

        // now we are ready to create our combined attributes and test if they work or not.
        // if our threadID is the seedblock, we can just skip

        // we don't have to test blocks before the seedblock, since they've already been tested
        // also we aren't going to have a block if the threadID is too big.
        if (threadID <= seedBlockNum || threadID >= numBlocks){}

        // make the combined mins and maxes, and then check against all our data.
        else{

            // first we build our combined list.
            for (int i = 0; i < numAttributes; i++){
                thisBlockCombinedMaxes[i] = max(seedBlockMaxes[i], hyperBlockMaxes[threadID * numAttributes + i]);
                thisBlockCombinedMins[i] = min(seedBlockMins[i], hyperBlockMins[threadID * numAttributes + i]);
            }

            // now we check all our data for a point falling into our new bounds.
            char allPassed = 1;
            for (int point = 0; point < numPoints; point++){

                char someAttributeOutside = 0;
                for(int att = 0; att < numAttributes; att++){
                    if (points[point * numAttributes + att] > thisBlockCombinedMaxes[att] || points[point * numAttributes + att] < thisBlockCombinedMins[att]){
                        someAttributeOutside = 1;
                        break;
                    }
                }
                // if there's NOT some attribute outside, this point has fallen in, and we can't do the merge.
                if (!someAttributeOutside){
                    allPassed = -1;
                    break;
                }
            }
            // if we didn't pass, we simply do nothing. if we did pass all the points, that means we can merge, and we can set the updated mins and maxes for this point to be the combined attributes instead.
            // then we simply flag that seedBlock is trash.
            if (allPassed){
                // copy the combined mins and maxes into the original array
                for (int i = 0; i < numAttributes; i++){
                    hyperBlockMins[threadID * numAttributes + i] = thisBlockCombinedMins[threadID * numAttributes + i];
                    hyperBlockMaxes[threadID * numAttributes + i] = thisBlockCombinedMaxes[threadID * numAttributes + i];
                }
                // set the flag to -1. atomic because many threads will try this.
                atomicMin(&deleteFlags[seedBlockNum], -1);
            }
            // sync threads to get ready at the end for the next iteration with the next seedBlock

        } // checking one seedblock loop
        __syncthreads();
    }
}
*/

/*
__global__ void MergerHelper1(double *seedHBMax, double *seedHBMin, double *mergingHBMaxes, double *mergingHBMins, double *combinedMax, double *combinedMin, double *opClassPnts, int *toBeDeleted, int numDims, int numMergingHBs, int cases)
{
    // Get the thread that we are on
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    do{
        if (threadID < numMergingHBs)
        {
            int offset = threadID * numDims;

            // go through all attributes.
            for (int i = 0; i < numDims; i++)
            {
                // Make a "mega-block!!!!"
                combinedMax[i+offset] = max(seedHBMax[i], mergingHBMaxes[i+offset]);
                combinedMin[i+offset] = min(seedHBMin[i], mergingHBMins[i+offset]);
            }

            // 1 = do merge, 0 = do not merge
            char merge = 1;

            // Go through all points, check if other class points will fall in
            for (int i = 0; i < cases; i += numDims)
            {
                bool withinSpace = true;
                for (int j = 0; j < numDims; j++)
                {
                    if (!(opClassPnts[i+j] <= combinedMax[j+offset] && opClassPnts[i+j] >= combinedMin[j+offset]))
                    {
                        withinSpace = false;
                        break;
                    }
                }

                if (withinSpace)
                {
                    merge = 0;
                    break;
                }
            }

            // If no points fell in we can remove the one that was merging with the seed.
            toBeDeleted[threadID] = merge;
        }




        // Sync the threads to see if any action was taken this time around
        __syncthreads();
    }
    while(actionTaken || cnt > 0)
}
*/