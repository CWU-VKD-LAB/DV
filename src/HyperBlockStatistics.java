import java.lang.reflect.Array;
import java.util.*;
import java.util.stream.Collectors;


//TODO: Add HBs per class to statistics and comparisons
//TODO: Add # of clauses per class to clause portion of comparison
//TODO: Add attributes per HB
//TODO: SIMPLIFY PORTIONS WHERE IT SAYS WHICH ATTRIBUTES ARE PRESENT/REMOVED EX: {x1-x4, x6-x7}
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

    /**
     * Record for statistics. Could be good idea to break up, but not doing that yet.
     * @param totalDataPoints Total number of datapoints in the dataset loaded.
     * @param numBlocks The current number of blocks.
     * @param numBlocksPerClass The number of blocks per class.
     * @param totalInBlocks The total number of points in at least 1 hyper-block. (no double counting)
     * @param coverage The % coverage, meaning % of points that belong to at least 1 block.
     * @param usedAttributes The attributes that were used for classification across whole DS.
     * @param clauseCountsHBs The clause counts of each of the current HyperBlocks.
     * @param totalClauses The total number of clauses used. ex: ".5 < x1 < 1" this is ONE clause.
     * @param algoLog The log of which simplifications have been run on the blocks.
     * @param classClauseCounts The clause counts by class.
     * @param averageSizeByClass The average number of points in blocks of each class.
     * @param averageHBSize The average number of points per hyper-block.
     * @param numSmallHBs The number of hyper-blocks with fewer than some threshold number of points.
     * @param nonDistinctPointCounts The number of non-distinct points each HB holds. (ALLOWS DOUBLE COUNTING OF POINTS)
     * @param usedAttributesByClass The attributes that were used to classify each class.
     * @param usedAttributesPerBlock The attributes that were used to classify each block.
     */
    public record statisticSet(int totalDataPoints, int numBlocks, int[] numBlocksPerClass, int totalInBlocks, double coverage,
                               ArrayList<Integer> usedAttributes, int[] clauseCountsHBs, int totalClauses,
                               ArrayList<String> algoLog, int[] classClauseCounts, double[] averageSizeByClass,
                               double averageHBSize, int numSmallHBs, int[] nonDistinctPointCounts, ArrayList<ArrayList<Integer>> usedAttributesByClass,
                               ArrayList<ArrayList<Integer>> usedAttributesPerBlock
    ){}

    private ArrayList<statisticSet> statisticHistory;

    boolean debug = true;

    public HyperBlockStatistics(HyperBlockGeneration hbGen){
        this.hbGen = hbGen;
        this.data = hbGen.data;
        this.hyper_blocks = hbGen.hyper_blocks;

        // Make a new statistic history.
        this.statisticHistory = new ArrayList<>();

        // Generate stats for the initially generated blocks.
        updateHyperBlockStatistics();
        //auto();
    }

    public void auto(){
        // Do disjunctive
        hbGen.simplifyHBtoDisjunctiveForm();
        hbGen.simplificationAlgoLog.add("Create Disjunctive Blocks");
        updateHyperBlockStatistics();
        //compareStatistics(0,1);

        //autoReset();
        hbGen.removeUselessAttributes();
        hbGen.simplificationAlgoLog.add("Remove Useless Attributes");
        updateHyperBlockStatistics();
        //compareStatistics(0,1);

        //autoReset();
        hbGen.removeUselessBlocks();
        hbGen.simplificationAlgoLog.add("Remove Useless Blocks");
        updateHyperBlockStatistics();
        compareStatistics(0,3);
    }

    private void autoReset(){
        // Clear the algo log in hb generation
        hbGen.simplificationAlgoLog.clear();
        // Make the default blocks again (COULD CHANGE THIS TO JUST STORE INITIAL ONES)
        hbGen.generateHBs(true);
        // Clear stats sets in here
        statisticHistory.clear();
        // generate for default blocks.
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
    private void compareStatistics(int before, int after) {
        System.out.println(statisticHistory.size() - 1 + "is the last index of the set");
        for(int i = 0; i < statisticHistory.size(); i++){
            System.out.println(i + " : " + statisticHistory.get(i).algoLog);
        }
        if (debug) {
            Scanner scan = new Scanner(System.in);
            System.out.println("Before: ");
            before = scan.nextInt();
            System.out.println("After: ");
            after = scan.nextInt();
        }

        // Set for the before info and set for the after info
        statisticSet sBefore = statisticHistory.get(before);
        statisticSet sAfter = statisticHistory.get(after);

        System.out.println("=== DATASET INFO ===");
        System.out.printf("\t%s\n", DV.dataFileName);
        System.out.printf("\t%s-D\n", DV.fieldLength);
        System.out.printf("\t%s points\n\n", sBefore.totalDataPoints);


        // Difference in the simplifications ran from setB to setA.
        System.out.println("=== SIMPLIFICATIONS FROM BEFORE TO AFTER ===");
        for (int i = sBefore.algoLog.size(); i < sAfter.algoLog.size(); i++) {
            System.out.println("\t" + (i + 1) + ". " + sAfter.algoLog.get(i));
        }
        // ex before "Remove Useless" after "Create Disjunctive"

        // % reduction in blocks. 10 -> 8
        double blockChange = 100 - ((double) sAfter.numBlocks / sBefore.numBlocks) * 100.0;
        double smallBlockChange = 100 - ((double) sAfter.numSmallHBs / sBefore.numSmallHBs) * 100.0;
        double averageBlockSizeChange = -(100 - ((double) sAfter.averageHBSize / sBefore.averageHBSize) * 100.0);

        System.out.println("\n=== BLOCKS ===");
        System.out.printf("\t%-25s %d ---> %d\n", "Total number blocks :", sBefore.numBlocks, sAfter.numBlocks);
        System.out.printf("\t%.2f%% fewer blocks.\n\n", blockChange);
        System.out.println("Coverage: " + sAfter.coverage);
        System.out.println("=== BLOCKS BY CLASS ===");

        // Print these numbers by class
        for (int i = 0; i < DV.classNumber; i++) {
            System.out.printf("\t%-25s Class: \"%s\" %d ---> %d\n", "Total number blocks - ", DV.uniqueClasses.get(i), sBefore.numBlocksPerClass[i], sAfter.numBlocksPerClass[i]);
            double temp = 100 - ((double) sAfter.numBlocksPerClass[i] / sBefore.numBlocksPerClass[i]) * 100.0;
            System.out.printf("\t%.2f%% fewer blocks.\n\n", temp);
        }

        // Number of small blocks:
        System.out.printf("\t%-25s %d ---> %d\n", "Number small blocks :", sBefore.numSmallHBs, sAfter.numSmallHBs);
        System.out.printf("\t%.2f%% fewer small blocks.\n\n", smallBlockChange);

        // AVERAGE BLOCK SIZES, AND BY CLASS SIZES
        System.out.printf("\t%-25s %.2f ---> %.2f\n", "Avg. points/block :", sBefore.averageHBSize, sAfter.averageHBSize);
        System.out.printf("\t%.2f%% more avg. points/block\n\n", averageBlockSizeChange);

        for (int i = 0; i < DV.classNumber; i++) {
            System.out.printf("\tClass \"%s\" : %f ---> %f avg. points/block.\n", DV.uniqueClasses.get(i), sBefore.averageSizeByClass[i], sAfter.averageSizeByClass[i]);
        }

        System.out.println("\n=== CLAUSES ===");
        // % reduction in clauses.
        double clauseChange = 100 - ((double) sAfter.totalClauses / sBefore.totalClauses) * 100.0;
        System.out.printf("\t%d clauses ---> %d clauses.\n", sBefore.totalClauses, sAfter.totalClauses);

        System.out.printf("\t%.2f%% fewer clauses total.\n\n", clauseChange);

        // % reduction in clauses per class
        double[] classClauseChanges = new double[sAfter.classClauseCounts.length];
        for (int i = 0; i < sAfter.classClauseCounts.length; i++) {
            classClauseChanges[i] = 100 - ((double) sAfter.classClauseCounts[i] / sBefore.classClauseCounts[i]) * 100.0;
            System.out.printf("\tClass \"%s\" number of clauses:   %d before ---> %d after.", DV.uniqueClasses.get(i), sBefore.classClauseCounts[i], sAfter.classClauseCounts[i]);
            System.out.printf("\tClass \"%s\": %.2f%% fewer clauses.\n", DV.uniqueClasses.get(i), classClauseChanges[i]);
        }

        double attributeChange = 100 - ((double) sAfter.usedAttributes.size() / sBefore.usedAttributes.size()) * 100.0;
        System.out.println("\n=== Attributes ===");
        System.out.printf("\t%d ---> %d attributes used.  (-%.2f%%)\n", sBefore.usedAttributes.size(), sAfter.usedAttributes.size(), attributeChange);

        System.out.print("\tAttributes used before: ");
        List<Integer> attributes = sBefore.usedAttributes;
        printIntervalCondensed(attributes);


        System.out.print("\n\tAttributes used after: ");
        attributes = sAfter.usedAttributes;
        printIntervalCondensed(attributes);


        System.out.print("\n\tThe attributes removed from before to after were: ");
        ArrayList<Integer> removed = new ArrayList<>(sBefore.usedAttributes);
        removed.removeAll(sAfter.usedAttributes);
        if (removed.isEmpty()) {
            System.out.print("NONE. \n");
        }else{
            printIntervalCondensed(removed);
        }


        System.out.print("\n=== ATTRIBUTES BY CLASS ===");

        for (int i = 0; i < DV.classNumber; i++) {
            // Print "Before" line
            System.out.printf("\n\t%-6s : %-14s", "Before - Class ", DV.uniqueClasses.get(i));
            printIntervalCondensed(sBefore.usedAttributesByClass.get(i));

            // Print "After" line
            System.out.printf("\n\t%-7s : %-15s", "After - Class ", DV.uniqueClasses.get(i));
            printIntervalCondensed(sAfter.usedAttributesByClass.get(i));

        }

        System.out.println("\n=== ATTRIBUTES BY BLOCK ===");

        System.out.print("\t= BEFORE BLOCKS = ");
        for (int i = 0; i < sBefore.usedAttributesPerBlock.size(); i++) {
            int n = sBefore.usedAttributesPerBlock.get(i).size() - 1;
            int last = sBefore.usedAttributesPerBlock.get(i).get(n);
            System.out.printf("\n\tBlock #%d from class \"%s\" :", sBefore.usedAttributesPerBlock.get(i).get(n-1), DV.uniqueClasses.get(last));

            // Condensed print all but last 2 elements of each block row.
            printIntervalCondensed(sBefore.usedAttributesPerBlock.get(i).subList(0, n-1));
        }

        System.out.print("\n\n\t= AFTER BLOCKS = ");
        int size = sBefore.usedAttributesPerBlock.size();

        // Create set of unique IDs present in sAfter for quick lookup
        Set<Integer> afterBlockIds = sAfter.usedAttributesPerBlock.stream()
                .map(block -> block.get(block.size() - 2)) // Get unique ID
                .collect(Collectors.toSet());

        // Iterate through all blocks (0 to size - 1)
        for (int i = 0; i < size; i++) {
            if (afterBlockIds.contains(i)) {
                // Block exists in sAfter, retrieve it
                int finalI = i;
                ArrayList<Integer> block = sAfter.usedAttributesPerBlock.stream()
                        .filter(b -> b.get(b.size() - 2) == finalI)
                        .findFirst()
                        .orElse(null);

                if (block != null) {
                    int n = block.size() - 1;
                    int classIndex = block.get(n);
                    String className = (classIndex >= 0 && classIndex < DV.uniqueClasses.size())
                            ? DV.uniqueClasses.get(classIndex)
                            : "Unknown";

                    System.out.printf("\n\tBlock #%d from class \"%s\" :", i, className);
                    printIntervalCondensed(block.subList(0, n - 1));
                }
            } else {
                // If block was removed.
                System.out.printf("\n\tBlock #%d is DELETED.", i);
            }
        }

        System.out.print("\n\n= ATTRIBUTES REMOVED BETWEEN BEFORE/AFTER BY BLOCK =");
        for (ArrayList<Integer> afterBlock : sAfter.usedAttributesPerBlock) {
            // Extract unique ID and class from the afterBlock
            int uniqueId = afterBlock.get(afterBlock.size() - 2);
            int bClass = afterBlock.get(afterBlock.size() - 1);

            // Find the corresponding block in sBefore
            ArrayList<Integer> beforeBlock = sBefore.usedAttributesPerBlock.stream()
                    .filter(block -> block.get(block.size() - 2) == uniqueId)
                    .findFirst()
                    .orElse(null);

            if (beforeBlock != null) {
                // Extract attributes (excluding last two elements)
                Set<Integer> beforeAttributes = new HashSet<>(beforeBlock.subList(0, beforeBlock.size() - 2));
                Set<Integer> afterAttributes = new HashSet<>(afterBlock.subList(0, afterBlock.size() - 2));

                // Find removed attributes (present in before but not in after)
                beforeAttributes.removeAll(afterAttributes);

                // Convert the Set to a List
                List<Integer> removedAttributes = new ArrayList<>(beforeAttributes);

                // Print block ID, class, and removed attributes
                System.out.printf("\n\tBlock #%d (Class %s) attributes that were removed: ", uniqueId, DV.uniqueClasses.get(bClass));
                printIntervalCondensed(removedAttributes);
            }
        }

        System.out.println("\n");
    }

    private void printIntervalCondensed(List<Integer> attributes) {
        if (attributes.isEmpty()) {
            System.out.print("None");
        } else {
            // Sort the attributes list to ensure it's in order
            Collections.sort(attributes);

            int start = attributes.get(0);
            int previous = start;

            for (int i = 1; i < attributes.size(); i++) {
                int current = attributes.get(i);
                if (current != previous + 1) {
                    // Print the range or single value
                    if (start == previous) {
                        System.out.printf("x%d, ", start);
                    } else {
                        System.out.printf("x%d-x%d, ", start, previous);
                    }
                    // Update the start of the next range
                    start = current;
                }
                previous = current;
            }

            // Print the final range or single value
            if (start == previous) {
                System.out.printf("x%d", start);
            } else {
                System.out.printf("x%d-x%d", start, previous);
            }
        }
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

        compareStatistics(0, 1);
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
        int[] numBlocksPerClass = findNumBlocksPerClass();
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

        int numSmallHBs = numberOfSmallBlocks(nonDistinctPointCounts, 5);
        ArrayList<ArrayList<Integer>> usedAttributesByClass = findImportantAttributesForClasses();
        ArrayList<ArrayList<Integer>> usedAttributesPerBlock = attributesPerHB();


        //averageSizeByClass, averageHBSize, numSmallHBs, nonDistinctPointCounts, usedAttributesByClass
        // Sorry.
        return new statisticSet(totalDataPoints, numBlocks,numBlocksPerClass, totalInBlocks, coverage,
                usedAttributes,clauseCountsHBs, totalClauses, algoLog, classClauseCounts,
                averageSizeByClass, averageHBSize, numSmallHBs, nonDistinctPointCounts, usedAttributesByClass, usedAttributesPerBlock);
    }


    /**
     * Finds the number of hyper-blocks per class.
     * @return int[] each index is the count per the corresponding class.
     */
    private int[] findNumBlocksPerClass() {
        int[] numBlocksPerClass = new int[DV.classNumber];

        // Go through all the hypeblocks and increment the count per class
        for(HyperBlock hb : hyper_blocks){
            numBlocksPerClass[hb.classNum]++;
        }

        return numBlocksPerClass;
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

    /**
     * Last element in each row is the class number of the block
     * 2nd last element in each row is the originalPosition of the block.
     * @return
     */
    private ArrayList<ArrayList<Integer>> attributesPerHB(){
        ArrayList<ArrayList<Integer>> hbList = new ArrayList<>();

        // Go through all the hyperblocks.
        for(int hb = 0; hb < hyper_blocks.size(); hb++){
            HyperBlock block = hyper_blocks.get(hb);
            hbList.add(new ArrayList<>());

            for(int i = 0; i < DV.fieldLength; i++){
                // If either one has a non-useless boundary add the current attribute to the list.
                if(block.maximums.get(i).get(0) == 1 && block.minimums.get(i).get(0) == 0){
                    continue;
                }
                hbList.get(hb).add(i);
            }

            // Add the original position and class of the block as the last two elements
            hbList.get(hb).add(block.originalPosition);
            hbList.get(hb).add(block.classNum);
        }

        // Sort the list by the second-to-last element (original position)
        hbList.sort((row1, row2) -> {
            int val1 = row1.get(row1.size() - 2); // Second-to-last value of row1
            int val2 = row2.get(row2.size() - 2); // Second-to-last value of row2
            return Integer.compare(val1, val2); // Sort in ascending order
        });

        // Print for debugging purposes
        for(ArrayList<Integer> list: hbList){
            System.out.println(list);
        }

        return hbList;
    }
}
