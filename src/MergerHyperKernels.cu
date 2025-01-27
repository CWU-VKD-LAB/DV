extern "C"

#define max(a, b) (a > b)? a: b
#define min(a, b) (a > b)? b: a

// our basic procedure is this. start with block 0 as a seed. try and merge all the other guys to block 0. if anyone does, then we can mark 0 for deletion.
// iterate to the next one, merging the blocks to that one and so on.
// we have only passed in countercases, not cases of our own class, all the blocks we are testing are also already the same class.
// may need some changing around, when we test with a dataset over 2000 ish attributes we can't use shared memory to store the seedblock attributes, so we just will use global if such a thing happens.
__global__ void MergerHelper1(float *hyperBlockMins, float *hyperBlockMaxes, float *combinedMins, float *combinedMaxes, char *deleteFlags, int numAttributes, float *points, int numPoints, int numBlocks){

    // shared memory is going to be 2 * numAttributes * sizeof(float) long. this lets us load the whole seed block into memory
    extern shared float seedBlockAttributes[];

    // stored in the same array, but using two pointers we can make this easy.
    float *seedBlockMins = &seedBlockAttributes[0];
    float *seedBlockMaxes = &seedBlockAttributes[numAttributes];

    // get our thread id
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    float *thisBlockCombinedMins = &combinedMins[threadID * numAttributes];
    float *thisBlockCombinedMaxes = &combinedMaxes[threadID * numAttributes];

    for(int seedBlock = 0; seedBlock < numBlocks; seedBlock++){

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
        if (threadID > seedBlock && threadID < numBlocks){

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
            // if we did pass all the points, that means we can merge, and we can set the updated mins and maxes for this point to be the combined attributes instead.
            // then we simply flag that seedBlock is trash.
            if (allPassed){
                // copy the combined mins and maxes into the original array
                for (int i = 0; i < numAttributes; i++){
                    hyperBlockMins[threadID * numAttributes + i] = thisBlockCombinedMins[threadID * numAttributes + i];
                    hyperBlockMaxes[threadID * numAttributes + i] = thisBlockCombinedMaxes[threadID * numAttributes + i];
                }
                // set the flag to -1. atomic because many threads will try this.
                atomicMin(&deleteFlags[seedBlock], -1);
            }
        } // checking one seedblock loop
        __syncthreads();
    }
}
