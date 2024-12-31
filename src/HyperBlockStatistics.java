import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;

/**
 * This class will be the window that shows an in depth view into the statistics of the hyperblocks.
 *
 * 1. Clause counting statistics, (maybe an automated running of simplification algos to show user how effective they are)
 * 2. Total coverage of the data points.
 * 3. ...
 */
public class HyperBlockStatistics {
    private ArrayList<HyperBlock> hyper_blocks;
    private ArrayList<DataObject> data;

    public HyperBlockStatistics(ArrayList<HyperBlock> hyper_blocks, ArrayList<DataObject> data){
        this.hyper_blocks = hyper_blocks;
        this.data = data;

        //TODO: Create a solid design of what the window should look like.
        //Use this spot to build the gui.
        dataAnalytics();

    }

    private void dataAnalytics(){
        printClauseNumbers();

        System.out.println("Total number of blocks is " + hyper_blocks.size());

        int totalDataPoints = totalDataSetSize();
        System.out.println("SIZE OF THE DATASET IS  " + totalDataPoints + " POINTS");

        int totalInBlocks = totalPointsInABlock();
        System.out.println("TOTAL NUMBER THAT IS IN THE BLOCKS IS " + totalInBlocks);
    }

    /**
     * Helper for future HyperBlock statistics.
     * @return The number of points in the dataset that is currently loaded.
     */
    private int totalDataSetSize(){
        int totalDataPoints = 0;

        for(int i = 0; i < data.size(); i++){
            //System.out.println("CLASS " + i + " SIZE: " + data.get(i).data.length);
            totalDataPoints += data.get(i).data.length;
        }

        return totalDataPoints;
    }

    /**
     * Helper for future HyperBlock statistics.
     * @return The number of points that fall within a block. Each point is counted only once, so coverage is accurate.
     */
    private int totalPointsInABlock(){
        int totalInBlocks = 0;
        // Go through each class

        // Keep track of distinct points in each block.
        int[] in = new int[hyper_blocks.size()];

        for(int i = 0; i < data.size(); i++){
            // Go through each data point
            for(int j = 0; j < data.get(i).data.length; j++){
                double[] point = data.get(i).data[j];

                // Go through all blocks and let them claim a point
                for(int hb = 0; hb < hyper_blocks.size(); hb++){
                    // If it is inside a block, let them claim it and keep away from other blocks.

                    if(inside_HB(hb, point)){
                        in[hb]++;
                        totalInBlocks++;
                        break;
                    }

                }
            }
        }

        for(int i = 0; i < in.length; i++){
            System.out.println("BLOCK #" + (i+1) + " HAS " + in[i] + " POINTS");
        }

        System.out.println("TOTAL NUMBER THAT IS IN THE BLOCKS IS " + totalInBlocks);
        return totalInBlocks;
    }



    /**
     * //TODO: IDEALLY WE SHOULD JUST REUSE THIS FROM HYPERBLOCK GENERATION
     *
     * Checks if data is inside a hyper-block
     * @param hb1 Hyper-block number
     * @param data Data to check
     * @return True if data is inside hyper-block
     */
    private boolean inside_HB(int hb1, double[] data)
    {
        HyperBlock tempBlock = hyper_blocks.get(hb1);

        boolean inside = true;

        // Go through all attributes
        for (int i = 0; i < DV.fieldLength; i++)
        {
            boolean inAnInterval = false;

            // Go through all intervals the hyperblock allows for the attribute
            for(int j = 0; j < tempBlock.maximums.get(i).size(); j++){
                // If the datapoints value falls inside one of the intervals.
                if (data[i] >= tempBlock.minimums.get(i).get(j) && data[i] <= tempBlock.maximums.get(i).get(j)) {
                    inAnInterval = true;
                    break;
                }
            }

            if (!inAnInterval) {
                inside = false;
                break;
            }
        }

        // Should return true if the point is inside at least 1 interval for all attributes.
        return inside;
    }


    /**
     * Sums how many clauses are needed to identify each class throughout all hyper_blocks.
     * Prints num needed per class and num needed for the whole dataset.
     */
    private void printClauseNumbers(){
        // array for keeping track of clauses for each class
        int[] classCount = new int[DV.uniqueClasses.size()];
        for(HyperBlock block : hyper_blocks){
            for(int i = 0; i < DV.fieldLength; i++){
                // Range (0,1) means it's a useless attribute that won't be printed
                if(block.maximums.get(i).get(0) == 1 && block.minimums.get(i).get(0) == 0){
                    continue;
                }

                // Loops through all intervals of the current attribute and counts it.
                for(int j = 0; j < block.maximums.get(i).size(); j++){
                    classCount[block.classNum]++;
                }
            }
        }

        // Print numbers gathered
        for(int i = 0;  i < classCount.length; i++){
            System.out.println("TOTAL CLAUSES FOR CLASS {" + DV.uniqueClasses.get(i) + "}  :  " + classCount[i]);
        }
        System.out.println("TOTAL CLAUSES    :  "   + Arrays.stream(classCount).sum());
    }
}
