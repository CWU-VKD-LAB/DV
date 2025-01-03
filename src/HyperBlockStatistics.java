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
    private HyperBlockGeneration hbGen;

    public record statisticSet(int totalDataPoints, int numBlocks, int totalInBlocks, double coverage,
                               ArrayList<Integer> usedAttributes, int[] clauseCountsHBs, int totalClauses,
                               ArrayList<String> algoLog, int[] classClauseCounts, double[] averageSizeByClass,
                               double averageHBSize, int numSmallHBs, int[] nonDistinctPointCounts, ArrayList<ArrayList<Integer>> usedAttributesByClass){}

    private ArrayList<statisticSet> statisticHistory;

    boolean debug = false;

    public HyperBlockStatistics(HyperBlockGeneration hbGen){
        this.hbGen = hbGen;
        this.data = hbGen.data;
        this.hyper_blocks = hbGen.hyper_blocks;

        // Make a new statistic history.
        this.statisticHistory = new ArrayList<>();

        // Generate stats for the initially generated blocks.
        updateHyperBlockStatistics();
    }

    //TODO: IMPLEMENT A GUI WINDOW THAT WILL BE BUILT FOR WHEN THE USER FINALLY CLICKS TO SEE THE STATISTICS.
    public void statisticsGUI(){
    }

    /**
     * Compares two sets of statistics to find the differences.
     * The comparison will be expressed as a single set of numbers
     * that indicates the change to the different stats after the "after" simplification has happened.
     * Simply put: result =
     * @param before The statistic index to be considered the "before"
     * @param after The statistics index to be considered the "after"
     */
    private void compareStatistics(int before, int after){

    }

    public void consoleStatistics(){

        System.out.println("\n\n=======================NOW PRINTING ALL STATISTICS=======================");
        System.out.println("=========================================================================\n\n");

        for(statisticSet stats : statisticHistory){
            System.out.println("\n\n=================== HYPERBLOCK STATISTICS ===================\n");

            System.out.println("=== DATASET INFO ===");
            System.out.println("Current Dataset: " + DV.dataFileName);
            System.out.println("DATASET DIMENSIONALITY: " + DV.fieldLength);
            System.out.println("SIZE OF THE DATASET IS  " + stats.totalDataPoints + " POINTS");

            System.out.println("\n=== BLOCKS & COVERAGE ===");
            System.out.println("TOTAL NUMBER OF BLOCKS " + stats.numBlocks);
            System.out.println("NUMBER OF POINTS IN A BLOCK " + stats.totalInBlocks);
            System.out.println("TOTAL COVERAGE OF POINTS BY BLOCKS IS " + stats.coverage + "%");

            System.out.println("AVERAGE NUMBER OF POINTS IN BLOCKS: " + stats.averageHBSize);
            System.out.println("AVERAGE NUMBER OF POINTS IN BLOCKS BY CLASS:");
            for(int i = 0; i < DV.classNumber; i++){
                System.out.printf("\tCLASS %s: %.2f POINTS\n", DV.uniqueClasses.get(i), stats.averageSizeByClass[i]);
            }

            System.out.println("NUMBER OF SMALL BLOCKS:  " + stats.numSmallHBs);

            System.out.println("\n=== USED ATTRIBUTES ===");
            System.out.println("THE ATTRIBUTES THAT WERE USED WERE " + stats.usedAttributes);

            System.out.println("\n== BY CLASS ==");
            for(int i =  0; i < stats.usedAttributesByClass.size(); i++){
                System.out.println("\tCLASS: " + DV.uniqueClasses.get(i));
                System.out.println(stats.usedAttributesByClass.get(i) + "\n");
            }

            System.out.println("\n=== CLAUSE COUNTS ===");
            System.out.println("TOTAL CLAUSES IS "   + stats.totalClauses);
            System.out.println("BLOCK CLAUSE COUNTS " + Arrays.toString(stats.clauseCountsHBs));

            System.out.println("\n== BY CLASS ==");
            for(int i = 0; i < stats.classClauseCounts.length; i++){
                System.out.println("\tCLASS \"" + DV.uniqueClasses.get(i) + "\" : " + stats.classClauseCounts[i]);
            }

            System.out.println("\n=== SIMPLIFICATIONS ===");
            for(int i = 0; i < stats.algoLog.size(); i++){
                System.out.printf("\t%d. %s\n", i + 1, stats.algoLog.get(i));
            }

            //averageSizeByClass, nonDistinctPointCounts
            System.out.println("\n=================== END STATISTICS ===================\n");
        }
    }

    /**
     * This is the "endpoint" that should be called after a simplification is run in the HyperBlockGeneration class
     * this will generate statistics for the change and add it to the history.
     * Once user opens the statistics page this will be filled.
     */
    public void updateHyperBlockStatistics(){
        statisticHistory.add(dataAnalytics());
    }

    private statisticSet dataAnalytics(){

        int totalDataPoints = totalDataSetSize();
        int numBlocks = hyper_blocks.size();
        int totalInBlocks = totalPointsInABlock();
        double coverage = ((double) totalInBlocks / totalDataPoints) * 100;
        // Print out which attributes across dataset were important for classification.
        ArrayList<Integer> usedAttributes = findImportantAttributesTotal();
        int[] clauseCountsHBs = calculateBlockClauseCount();
        int totalClauses = Arrays.stream(clauseCountsHBs).sum();
        ArrayList<String> algoLog = new ArrayList<>(hbGen.simplificationAlgoLog);

        int[] classClauseCounts = calculateClassClauseCount();
        int[] nonDistinctPointCounts = numberOfNonDistinctPointsInBlocks();
        double[] averageSizeByClass = averageBlockSizeByClass(nonDistinctPointCounts);
        double averageHBSize = averageBlockSize(nonDistinctPointCounts);

        int numSmallHBs = numberOfSmallBlocks(nonDistinctPointCounts, (int) ((totalDataPoints * .02)));
        ArrayList<ArrayList<Integer>> usedAttributesByClass = findImportantAttributesForClasses();

        //averageSizeByClass, averageHBSize, numSmallHBs, nonDistinctPointCounts, usedAttributesByClass
        // Sorry.
        return new statisticSet(totalDataPoints, numBlocks, totalInBlocks, coverage,
                usedAttributes,clauseCountsHBs, totalClauses, algoLog, classClauseCounts,
                averageSizeByClass, averageHBSize, numSmallHBs, nonDistinctPointCounts, usedAttributesByClass);
    }

    /**
     * Helper for future HyperBlock statistics.
     * @return The number of points in the dataset that is currently loaded.
     */
    private int totalDataSetSize(){
        int totalDataPoints = 0;

        for(int i = 0; i < data.size(); i++){
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
     * Finds how many clauses there are for each class of hyper-block.
     * @return An int[] in which each element is the # of clauses for that hyper-block. arr[block.classNum]
     */
    private int[] calculateClassClauseCount(){
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

        return classCount;
    }

    /**
     * Gets the number of clauses for each individual Hyper-Block.
     * @return int[], counts[hb] is the # of clauses for block at hyper_blocks[hb]
     */
    private int[] calculateBlockClauseCount(){
        int[] clauseCounts = new int[hyper_blocks.size()];

        for(int hb = 0; hb < hyper_blocks.size(); hb++){
            HyperBlock block = hyper_blocks.get(hb);
            for(int i = 0; i < DV.fieldLength; i++){
                // Range (0,1) means it's a useless attribute that won't be printed
                if(block.maximums.get(i).get(0) == 1 && block.minimums.get(i).get(0) == 0){
                    continue;
                }

                // Loops through all intervals of the current attribute and counts it.
                for(int j = 0; j < block.maximums.get(i).size(); j++){
                    clauseCounts[hb]++;
                }
            }
        }

        return clauseCounts;
    }

    /**
     * This function should identify which attributes were used across all blocks.
     * If an attribute has a range of 0-1 across all blocks it should not be included.
     * @return
     */
    private ArrayList<Integer> findImportantAttributesTotal(){
        //
        boolean[] used = new boolean[DV.fieldLength];

        for(HyperBlock block : hyper_blocks){
            for(int i = 0; i < DV.fieldLength; i++){
                if(block.maximums.get(i).get(0) == 1 && block.minimums.get(i).get(0) == 0){
                    continue;
                }

                used[i] = true;
            }
        }

        ArrayList<Integer> usedAttributes = new ArrayList<>();
        for(int i = 0; i < used.length; i++){
            if(used[i]){
                usedAttributes.add(i);
            }
        }

        return usedAttributes;
    }

    /**
     * This function should find the attributes that are important for each individual class.
     * For example Iris-Setosa may only need x3, so x3 would be the only one that is included.
     * @return ArrayList<ArrayList<Integer>>, each entry in outer list is for a class, inside will be the important attributes.
     */
    private ArrayList<ArrayList<Integer>> findImportantAttributesForClasses(){
        boolean[][] used = new boolean[DV.classNumber][DV.fieldLength];

        for(HyperBlock block : hyper_blocks){
            for(int i = 0; i < DV.fieldLength; i++){
                if(block.maximums.get(i).get(0) == 1 && block.minimums.get(i).get(0) == 0){
                    continue;
                }

                used[block.classNum][i] = true;
            }
        }

        ArrayList<ArrayList<Integer>> usedAttrs = new ArrayList<>();
        for(int i = 0; i < used.length; i++){
            usedAttrs.add(new ArrayList<>());
        }

        for(int i = 0; i < used.length; i++){
            for(int j = 0; j < used[i].length; j++){
                if(used[i][j]){
                    // Add the current attribute index to the list of used attributes.
                    usedAttrs.get(i).add(j);
                }
            }
        }
        return usedAttrs;
    }

    private int[] numberOfNonDistinctPointsInBlocks(){
        int[] pointCounts = new int[hyper_blocks.size()];

        // Go through all blocks and count points inside.
        for(int hb = 0; hb < hyper_blocks.size(); hb++){
            for(int i = 0; i < data.size(); i++){
                for(double[] point : data.get(i).data){
                    if(inside_HB(hb, point)){
                        pointCounts[hb]++;
                    }
                }
            }
        }

        return pointCounts;
    }

    /**
     * Should return the number of blocks that are "small" or overfit as categorized by threshold.
     * @param pointsInBlocks Arraylist with number of non-distinct points each block has.
     * @param threshold The threshold for a block to be considered a small block.
     * @return The integer number of blocks which have # points <= threshold.
     */
    private int numberOfSmallBlocks(int[] pointsInBlocks, int threshold){
        int count = 0;

        for(int num : pointsInBlocks){
            if(num <= threshold){
                count++;
            }
        }

        return count;
    }

    /**
     * Returns the average number of points in each block.
     * @param pointsInBlocks Array with # of points for each block.
     * @return Average number of points in all blocks.
     */
    private double averageBlockSize(int[] pointsInBlocks){
        return ((double)Arrays.stream(pointsInBlocks).sum() / pointsInBlocks.length);
    }

    /**
     * Returns the average number of points in each class of blocks.
     * @param pointsInBlocks Array with # of points for each block.
     * @return Average number of points in blocks of each class.
     */
    private double[] averageBlockSizeByClass(int[] pointsInBlocks){
        // Hold number of points, and number of blocks of each class.
        double[] sumsThenAvgs = new double[DV.classNumber];
        int[] counts = new int[DV.classNumber];

        for(int hb = 0; hb < hyper_blocks.size(); hb++){
            HyperBlock block = hyper_blocks.get(hb);
            int numP = pointsInBlocks[hb];

            sumsThenAvgs[block.classNum] += numP;
            counts[block.classNum]++;
        }

        // Convert each into an average.
        for(int i = 0; i < DV.classNumber; i++){
            sumsThenAvgs[i] = sumsThenAvgs[i] / counts[i];
        }

        return sumsThenAvgs;
    }

}
