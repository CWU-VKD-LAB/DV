extern "C"

#define max(a, b) (a > b)? a: b
#define min(a, b) (a > b)? b: a

// our basic procedure is this. start with block 0 as a seed. try and merge all the other guys to block 0. if anyone does, then we can mark 0 for deletion.
// iterate to the next one, merging the blocks to that one and so on.
// we have only passed in countercases, not cases of our own class, all the blocks we are testing are also already the same class.
// may need some changing around, when we test with a dataset over 2000 ish attributes we can't use shared memory to store the seedblock attributes, so we just will use global if such a thing happens.
__global__ void MergerHelper1(float *hyperBlockMins, float *hyperBlockMaxes, float *combinedMins, float *combinedMaxes, int *deleteFlags, int *mergable, const int numAttributes, float *points, const int numPoints, const int numBlocks, int* readSeedQueue, int* usedFlagsDebugOnly, int* blockSync, int totalThreadCnt, int* writeSeedQueue){
    //printf("Hello from inside the kernel.");
    // shared memory is going to be 2 * numAttributes * sizeof(float) long. this lets us load the whole seed block into memory
    extern __shared__ float seedBlockAttributes[];
    int numCudaBlocks = gridDim.x;
    //const int CUDA_NUM_BLOCKS = gridDim.x;

    // stored in the same array, but using two pointers we can make this easy.
    float *seedBlockMins = &seedBlockAttributes[0];
    float *seedBlockMaxes = &seedBlockAttributes[numAttributes];

    // get our thread id
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    //printf("%d", CUDA_NUM_BLOCKS);

    float *thisBlockCombinedMins = &combinedMins[threadID * numAttributes];
    float *thisBlockCombinedMaxes = &combinedMaxes[threadID * numAttributes];

    for(int deadSeedNum = 0; deadSeedNum < numBlocks; deadSeedNum++){
        if(threadID == 0){
            printf("%d\n", deadSeedNum);
        }
        __syncthreads();

        if(deadSeedNum > 0){
            int* tmp = readSeedQueue;
            readSeedQueue = writeSeedQueue;
            writeSeedQueue = tmp;
        }

        if(threadIdx.x == 0){atomicAdd(blockSync, 1);     while(atomicAdd(blockSync, 0) < numCudaBlocks) { }} // SPIN TO SYNC BLOCKS
        __syncthreads();

        if(threadID == 0){
            atomicExch(blockSync, 0); // Reset the global int to 0.
        }

        // This is the thing with accessing the right block from the queue.
        int seedBlock = readSeedQueue[deadSeedNum];

        //if(threadID == 0){usedFlagsDebugOnly[seedBlock] = 1;} //DEBUG

        __syncthreads();

        // copy the seed blocks attributes into shared memory, so that we can load stuff much faster.block Dim is how many threads in a block. since each block has it's own shared memory, this is our offset for copying stuff over. this is needed becausewe could have more than 1024 attributes very easily.
        for (int index = threadIdx.x; index < numAttributes; index += blockDim.x){
            seedBlockMins[index] = hyperBlockMins[seedBlock * numAttributes + index];
            seedBlockMaxes[index] = hyperBlockMaxes[seedBlock * numAttributes + index];
        }
        // sync when we're done copying over the seedblock values.
        __syncthreads();

        int k = threadID;

        while(k < numBlocks){

            // make the combined mins and maxes, and then check against all our data.
            if (k < numBlocks && k != seedBlock && deleteFlags[k] != -1 ){

                // first we build our combined list.
                for (int i = 0; i < numAttributes; i++){
                    thisBlockCombinedMaxes[i] = max(seedBlockMaxes[i], hyperBlockMaxes[k * numAttributes + i]);
                    thisBlockCombinedMins[i] = min(seedBlockMins[i], hyperBlockMins[k  * numAttributes + i]);
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
                if (allPassed){
                    // copy the combined mins and maxes into the original array
                    for (int i = 0; i < numAttributes; i++){
                        hyperBlockMins[k * numAttributes + i] = thisBlockCombinedMins[i];
                        hyperBlockMaxes[k * numAttributes + i] = thisBlockCombinedMaxes[i];
                    }
                    // set the flag to -1. atomic because many threads will try this.

                    atomicMin(&deleteFlags[seedBlock], -1);
                    mergable[k] = 1;
                }
            }

            // Move the threads to their new HyperBlock
            k += totalThreadCnt;
        }


        if(threadIdx.x == 0){atomicAdd(blockSync, 1); while(atomicAdd(blockSync, 0) < numCudaBlocks) { }} // SYNC BLOCKS
        __syncthreads();

        if(threadID == 0){
           atomicExch(blockSync, 0); // Reset the global int to 0.
           for(int i = 0; i <= deadSeedNum; i++){
               mergable[readSeedQueue[i]] = 0;
           }
        }

        __syncthreads();
        //Reset
        k = threadID;
        while(k < numBlocks){
            if(k > deadSeedNum){
                // Redo the order of the non-existent queue readSeedQueue
                int blockNum = readSeedQueue[k];
                int cnt = 0;

                if(mergable[blockNum] == 1){
                    // Count how many 1's are to the left of me, put self
                    for(int i = k - 1; i > deadSeedNum; i--){ // K should work, since it will have been at the k spot when thread read.
                        if(mergable[readSeedQueue[i]] == 1){
                            cnt++;
                        }
                    }
                    writeSeedQueue[numBlocks - 1 - cnt] = blockNum;
                }
                else{
                   cnt += deadSeedNum + 1;
                   // count all non-1's to the left, then put self in deadSeedNum + count
                   for(int i = k - 1; i > deadSeedNum; i--){
                        if(mergable[readSeedQueue[i]] == 0){
                            cnt++;
                        }
                   }
                   writeSeedQueue[cnt] = blockNum;
                }
            }
            k += totalThreadCnt;
        }


         if(threadIdx.x == 0){atomicAdd(blockSync, 1); while(atomicAdd(blockSync, 0) < numCudaBlocks) { }} // SYNC BLOCKS
         if(threadID == 0){
            atomicExch(blockSync, 0); // Reset the global int to 0.
         }

        __syncthreads();

        //RESET THE MERGING FLAGS
        k = threadID;
        while(k < numBlocks){
            mergable[k] = 0;
            k += totalThreadCnt;
        }
    }
}