import org.jfree.chart.*;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.axis.NumberTickUnit;
import org.jfree.chart.axis.ValueAxis;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.category.BarRenderer;
import org.jfree.chart.renderer.xy.XYAreaRenderer;
import org.jfree.chart.renderer.xy.XYBlockRenderer;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.category.DefaultCategoryDataset;
import org.jfree.data.xy.DefaultXYZDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.geom.Ellipse2D;
import java.awt.geom.Rectangle2D;
import java.awt.image.AreaAveragingScaleFilter;
import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.file.Files;
import java.sql.Array;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.List;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicIntegerArray;
import java.util.logging.Level;
import java.util.logging.Logger;
import jcuda.*;
import jcuda.driver.*;

import static jcuda.driver.JCudaDriver.*;


public class HyperBlockGeneration
{
    /************************************************************
     * Logging
     ************************************************************/
    private final static Logger LOGGER = Logger.getLogger(Analytics.class.getName());

    /************************************************************
     * Data
     ************************************************************/
    ArrayList<DataObject> data;

    /************************************************************
     * Hyperblocks
     ************************************************************/
    JSpinner hb_lvl;

    // hyperblocks
    ArrayList<HyperBlock> hyper_blocks = new ArrayList<>();
    ArrayList<ArrayList<HyperBlock>> hyper_blocks_levels = new ArrayList<>();

    // metrics
    ArrayList<ArrayList<double[]>> average_case = new ArrayList<>();
    ArrayList<Double> accuracy = new ArrayList<>();
    ArrayList<Integer> misclassified = new ArrayList<>();
    ArrayList<String> simplificationAlgoLog = new ArrayList<>();

    /************************************************************
     * General Line Coordinates Hyperblock Rules Linear (GLC-HBRL)
     ************************************************************/
    double acc_threshold = 100;
    boolean explain_ldf = false;
    int best_attribute = 0;

    /************************************************************
     * General User Interface
     ************************************************************/
    // panels 
    JPanel graphPanel;
    JPanel navPanel;
    JPanel expansionPanel;
    JScrollPane expansionScroll;
    // visualization options
    JLabel graphLabel;
    JButton right;
    JButton left;
    int visualized_block = 0;

    // initial plot choice
    int plot_id = 0;

    JComboBox<String> plotOptions;
    JComboBox<String> viewOptions;
    JComboBox<String> dataViewOptions;
    JComboBox<String> simplifications;
    JButton statistics;
    // colors
    Color[] graphColors;

    boolean remove_extra = false;

    record Interval(double start, double end) {}

    HyperBlockStatistics blockStats;

    HyperBlockGeneration() {
        // set purity threshold
        if (explain_ldf)
            acc_threshold = DV.accuracy;

        // find best attribute to sort data on
        findBestAttribute();

        // generate hyperblocks, and print hyperblock info
        getData();
        generateHBs(true);
        HB_analytics();

        // visualize hyperblocks
        if (!hyper_blocks.isEmpty())
        {
            HB_GUI();
            getPlot();
        }

        HB_analytics();

        /**
         * Set the original position of blocks in the list
         * This helps with keeping track of what blocks
         * were deleted in statistics.
         */
        for(int i = 0; i < hyper_blocks.size(); i++){
            HyperBlock hb = hyper_blocks.get(i);
            hb.originalPosition = i;
        }

        blockStats = new HyperBlockStatistics(this);
        // k-fold used to be here
        //test_HBs();
    }

    public void expandWithRealEdges(int hb_num, boolean[] attributes){
        System.out.println("Expanding with real edges.");
        int start = 0;
        int stop = hyper_blocks.size();

        if (hb_num > -1){
            start = hb_num;
            stop = hb_num + 1;
        }

        // Take the hyper-block we are currently on.
            //for maximums: find a point of same class that has closest value > the current max
                // try to expand to that value for the max
                    // check if block is still pure
            //for minimums: find a point of the same class that has closest value < the current min
                // try to expand
                    // check if the block is still pure.


        for(int hb = start; hb < stop; hb++){
            HyperBlock block = hyper_blocks.get(hb);

            for(int attr = 0; attr < attributes.length; attr++){
                // If they dont want to attempt to expand this attribute, skip.
                if(!attributes[attr]){
                    continue;
                }

                // We know they want to try to expand the attribute.
                ///
                for(int i = 0; i < block.maximums.get(attr).size(); i++){
                    // We should attempt to expand both sides of this interval.
                    double old_max = block.maximums.get(attr).get(i);

                    // The new block bounds should be a close

                    // Go through all the rows of the points that are in the same class
                    double new_max = Integer.MAX_VALUE;
                    for(int m = 0; m < data.get(block.classNum).data.length; m++){
                        double val = data.get(block.classNum).data[m][attr];
                        if(val > old_max && val < new_max){
                            new_max = val;
                        }
                    }

                    // This means we found no points to expand up to, go to next interval on this attribute.
                    if(new_max == Integer.MAX_VALUE){
                        continue;
                    }

                    // Set the interval
                    block.maximums.get(attr).set(i, new_max);
                    //block.minimums.get(attr).set(i, new_min);

                    // we now need to make sure the block is still pure.
                    ///////////////////////////////////////////////////////////////
                    // Go through all the data of other classes, check if it would fall into the new bounds of merged disjunctive block.
                    int classNum = block.classNum;
                    boolean doExpanse = true;
                    for(int k = 0; k < data.size(); k++) {

                        // skip data of same class
                        if(classNum == k){
                            continue;
                        }

                        for (double[] point : data.get(k).data) {
                            // Check if the point is within the MEGA BLOCK
                            if(inside_HB(hb, point)){
                                doExpanse = false;
                                break;
                            }
                        }

                        if(!doExpanse){break;}
                    }

                    // Undo the expanse on this interval.
                    if(!doExpanse) {
                        block.maximums.get(attr).set(i, old_max);
                    }
                }

                ///////// MINIMUMS PART
                for(int i = 0; i < block.minimums.get(attr).size(); i++){
                    // We should attempt to expand both sides of this interval.
                    double old_min = block.minimums.get(attr).get(i);

                    // The new block bounds should be a close

                    // Go through all the rows of the points that are in the same class
                    double new_min = Integer.MIN_VALUE;
                    for(int m = 0; m < data.get(block.classNum).data.length; m++){
                        double val = data.get(block.classNum).data[m][attr];
                        if(val < old_min && val > new_min){
                            new_min = val;
                        }
                    }
                    System.out.println("Attempting to downward expand to: " + new_min + " attribute: " + attr);

                    // This means we found no points to expand up to, go to next interval on this attribute.
                    if(new_min == Integer.MIN_VALUE){
                        continue;
                    }

                    // Set the interval
                    block.minimums.get(attr).set(i, new_min);

                    // we now need to make sure the block is still pure.
                    ///////////////////////////////////////////////////////////////
                    // Go through all the data of other classes, check if it would fall into the new bounds of merged disjunctive block.
                    int classNum = block.classNum;
                    boolean doExpanse = true;
                    for(int k = 0; k < data.size(); k++) {

                        // skip data of same class
                        if(classNum == k){
                            continue;
                        }

                        for (double[] point : data.get(k).data) {
                            // Check if the point is within the MEGA BLOCK
                            if(inside_HB(hb, point)){
                                doExpanse = false;
                                break;
                            }
                        }

                        if(!doExpanse){break;}
                    }

                    // Undo the expanse on this interval.
                    if(!doExpanse) {
                        block.minimums.get(attr).set(i, old_min);
                    }
                }
            }
        }
    }

    /**
     * Attempt to expand the intervals then check if any points of other classes would fall into the blocks.
     * @param amount The amount to try to expand the block by, ex .05, or .1
     * @param hb_num The index in hyper_blocks of the block to attempt to expand. PASS A NEGATIVE TO EXPAND ALL BLOCKS.
     */
    private void attemptToExpandIntervals(double amount, int hb_num, boolean keepEdgesReal, boolean[] attributes){
        if(keepEdgesReal){
            expandWithRealEdges(hb_num, attributes);
            return;
        }

        System.out.println("Made it past the keep edge real thingy.");
        int start = 0;
        int stop = hyper_blocks.size();

        if (hb_num > -1){
            start = hb_num;
            stop = hb_num + 1;
        }

        // Go through all blocks, or only the one the user entered index for.
        for(int hb = start; hb < stop; hb++){
            HyperBlock block = hyper_blocks.get(hb);
            // Go through all the attributes
            for(int attr = 0; attr < DV.fieldLength; attr++){
                // Go through all the intervals for the attribute
                for(int i = 0; i < block.maximums.get(attr).size(); i++){
                    // We should attempt to expand both sides of this interval.
                    double old_max = block.maximums.get(attr).get(i);
                    double old_min = block.minimums.get(attr).get(i);

                    // Handle the edge case of going over 1 or under 0
                    double max = Math.min(old_max + amount, 1);
                    double min = Math.max(old_min - amount, 0);

                    // Set the interval
                    block.maximums.get(attr).set(i, max);
                    block.minimums.get(attr).set(i, min);

                    // we now need to make sure the block is still pure.
                    ///////////////////////////////////////////////////////////////
                    // Go through all the data of other classes, check if it would fall into the new bounds of merged disjunctive block.
                    int classNum = block.classNum;
                    boolean doExpanse = true;
                    for(int k = 0; k < data.size(); k++) {

                        // skip data of same class
                        if(classNum == k){
                            continue;
                        }

                        for (double[] point : data.get(k).data) {
                            // Check if the point is within the MEGA BLOCK
                            if(inside_HB(hb, point)){
                                doExpanse = false;
                                break;
                            }
                        }

                        if(!doExpanse){break;}
                    }

                    // Undo the expanse on this interval.
                    if(!doExpanse) {
                        block.maximums.get(attr).set(i, old_max);
                        block.minimums.get(attr).set(i, old_min);
                    }
                }
            }
        }
    }


    /**
     * Goes through the existing hyper-blocks and tries to merge blocks of same class together.
     * The main goal of this function is to create disjunctive blocks. This means they can have OR cases for attribute intervals.
     */
    public void simplifyHBtoDisjunctiveForm(){
        //sort_hb_by_size();
        Set<Integer> blocksToBeRemoved = new HashSet<>();

        // Go through each hyperblock
        for(int i = 0; i < hyper_blocks.size(); i++){
            HyperBlock outerBlock = hyper_blocks.get(i);
            for(int j = 0; j < hyper_blocks.size(); j++){
                // Don't compare a block to itself.
                if(i == j){continue;}

                if (blocksToBeRemoved.contains(i)) break;

                if (blocksToBeRemoved.contains(j)) continue;

                // Get current inner block of same class
                HyperBlock innerBlock = hyper_blocks.get(j);

                if(outerBlock.classNum != innerBlock.classNum){
                    continue;
                }

                // First we want to check if the outer block is completely surrounding inner block. If it is,
                // we can remove the smaller block
                boolean allInside = true;
                for(int k = 0; k < DV.fieldLength; k++){

                    ArrayList<Double> outerBlockMinAtt = outerBlock.minimums.get(k);
                    ArrayList<Double> outerBlockMaxAtt = outerBlock.maximums.get(k);
                    ArrayList<Double> innerBlockMinAtt = innerBlock.minimums.get(k);
                    ArrayList<Double> innerBlockMaxAtt = innerBlock.maximums.get(k);

                    // OUTSIDE ARRAYLIST EACH ENTRY IS AN ATTRIBUTE: {{x1, x1.1, x1,2},{x2, x2.1, x2.1}, {}}

                    // iterate through all of x1 instances for inner block, and see if they are inside one of the outer block instances somewhere.
                    for (int innerAttributeInstance = 0; innerAttributeInstance < innerBlockMaxAtt.size(); innerAttributeInstance++) {
                        for (int outerAttributeInstance = 0; outerAttributeInstance < outerBlockMinAtt.size(); outerAttributeInstance++) {
                            // if inner min is less than outer min, or inner max is greater than outer max, we are not all inside.
                            if (innerBlockMinAtt.get(innerAttributeInstance) < outerBlockMinAtt.get(outerAttributeInstance) || innerBlockMaxAtt.get(innerAttributeInstance) > outerBlockMaxAtt.get(outerAttributeInstance)){
                                allInside = false;
                                break;
                            }
                        }
                    }
                }

                if(allInside){
                    // This means we can delete the smaller block.
                    blocksToBeRemoved.add(j);
                }else{

                    ArrayList<ArrayList<Double>> mins = new ArrayList<>();
                    ArrayList<ArrayList<Double>> maxes = new ArrayList<>();

                    // Create the MEGA BLOCK!!!!!!!
                    for(int k = 0; k < DV.fieldLength; k++){
                        mins.add(new ArrayList<>());
                        maxes.add(new ArrayList<>());

                        for(int curr = 0; curr < outerBlock.intervalCount(k); curr++){
                            mins.get(k).add(outerBlock.minimums.get(k).get(curr));
                            maxes.get(k).add(outerBlock.maximums.get(k).get(curr));
                        }

                        for(int curr = 0; curr < innerBlock.intervalCount(k); curr++){
                            mins.get(k).add(innerBlock.minimums.get(k).get(curr));
                            maxes.get(k).add(innerBlock.maximums.get(k).get(curr));
                        }

                        mergeIntervals(k, mins, maxes);
                    }

                    // Go through all the data of other classes, check if it would fall into the new bounds of merged disjunctive block.
                    int classNum = outerBlock.classNum;
                    boolean doMerge = true;
                    for(int k = 0; k < data.size(); k++) {

                        // skip data of same class
                        if(classNum == k){
                            continue;
                        }

                        for (double[] point : data.get(k).data) {
                            // Check if the point is within the MEGA BLOCK
                            if(inside_HB_Intervals(mins, maxes, point, -1)){
                                doMerge = false;
                                break;
                            }
                        }

                        if(!doMerge){break;}
                    }

                    if(doMerge) {
                        // Create deep copies of mins and maxes
                        ArrayList<ArrayList<Double>> copiedMins = new ArrayList<>();
                        ArrayList<ArrayList<Double>> copiedMaxes = new ArrayList<>();

                        for (ArrayList<Double> minList : mins) {
                            copiedMins.add(new ArrayList<>(minList));
                        }

                        for (ArrayList<Double> maxList : maxes) {
                            copiedMaxes.add(new ArrayList<>(maxList));
                        }

                        // Assign the copied mins and maxes to the hyper-block
                        outerBlock.minimums = copiedMins;
                        outerBlock.maximums = copiedMaxes;

                        // Mark the inner block for removal
                        blocksToBeRemoved.add(j);
                    }
                }
            }
        }

        // Remove them in reverse order so indices of blocks isn't shifted
        List<Integer> sortedBlocksToBeRemoved = new ArrayList<>(blocksToBeRemoved);
        sortedBlocksToBeRemoved.sort(Collections.reverseOrder());
        for(int i : sortedBlocksToBeRemoved){
            // SAFE WAY TO REMOVE A HYPER-BLOCK, WHILE UPDATING MIMIC FOR STATISTICS TRACKING
            hyper_blocks.remove(i);
        }
        order_hbs_by_class();
    }

    // Method to merge overlapping intervals for each attribute (k)
    private void mergeIntervals(int k, ArrayList<ArrayList<Double>> mins, ArrayList<ArrayList<Double>> maxes) {
        // Pair up the mins and maxes into interval objects for easy handling
        ArrayList<Interval> intervals = new ArrayList<>();
        for (int i = 0; i < mins.get(k).size(); i++) {
            intervals.add(new Interval(mins.get(k).get(i), maxes.get(k).get(i)));
        }

        // Sort the intervals based on their start (mins)
        intervals.sort(Comparator.comparingDouble(interval -> interval.start));

        // Merge overlapping intervals
        ArrayList<Interval> mergedIntervals = new ArrayList<>();
        Interval current = intervals.get(0);

        for (int i = 1; i < intervals.size(); i++) {
            Interval next = intervals.get(i);

            if (current.end >= next.start) {
                // If intervals overlap or are adjacent, merge them
                current = new Interval(current.start, Math.max(current.end, next.end));
            } else {
                // If no overlap, add the current interval to mergedIntervals
                mergedIntervals.add(current);
                current = next;
            }
        }
        // Add the last interval
        mergedIntervals.add(current);

        // Update the mins and maxes lists with the merged intervals
        mins.get(k).clear();
        maxes.get(k).clear();

        for (Interval interval : mergedIntervals) {
            if (interval.start == 0 && interval.end == 1.0){
                mins.get(k).clear();
                maxes.get(k).clear();
                mins.get(k).add(interval.start);
                maxes.get(k).add(interval.end);
                break;
            }
            mins.get(k).add(interval.start);
            maxes.get(k).add(interval.end);
        }
    }
    private void removeUselessAttributesCUDA(){
        // Initialize JCuda
        JCudaDriver.setExceptionsEnabled(true);
        cuInit(0);

        // Create a context
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);

        // Load the kernel
        CUmodule module1 = new CUmodule();
        cuModuleLoad(module1, ".\\src\\SimplificationsKernels.ptx");
        CUfunction removeUselessHelper = new CUfunction();
        cuModuleGetFunction(removeUselessHelper, module1, "removeUselessHelper");

    }
    public void removeUselessAttributes()
    {

        // Go through all hyperblocks
        for(int i = 0; i < hyper_blocks.size(); i++){
            HyperBlock tempBlock = hyper_blocks.get(i);

            int classNum = tempBlock.classNum;
            ArrayList<ArrayList<Double>> mins = new ArrayList<>(tempBlock.minimums);
            ArrayList<ArrayList<Double>> maxes = new ArrayList<>(tempBlock.maximums);

            // removed represents one particular attribute which we want to try and remove
            for (int removed = 0; removed < maxes.size(); removed++){

                //System.out.println("We are trying to remove attribute " + removed + " from hyperblock " + i + " which belongs to class " + tempBlock.classNum);
                boolean someoneInBounds = false;

                // iterating through all the OTHER classes, k representing a particular class number
                for(int k = 0; k < data.size(); k++){
                    // Skip going through data of same class as hyperblock
                    if(classNum == k){
                        continue;
                    }

                    // Go through each point in the class
                    for(double[] point : data.get(k).data){

                        boolean inside = inside_HB_Intervals(mins, maxes, point, removed);
                        if(inside){
                            someoneInBounds = true;
                            break;
                        }
                    }

                    if(someoneInBounds){
                        break;
                    }

                }

                // if nobody falls into our bounds, we can safely remove now.
                if(!someoneInBounds){
                    // Update the maxes/mins to allow all range [0, 1] aka removing attribute
                    maxes.get(removed).set(0, 1.0);
                    mins.get(removed).set(0, 0.0);

                    // Remove the other intervals, since the attribute got removed.
                    for(int f = mins.get(removed).size() - 1; f > 0; f--){
                        mins.get(removed).remove(f);
                        maxes.get(removed).remove(f);
                    }

                }
            }
            //TODO:AUSTIN DEEP COPY THIS MAYBE, OR JUST ACCESS IT DIRECTLY
            tempBlock.minimums = mins;
            tempBlock.maximums = maxes;
        }
    }

    /**
     * Create hyperblocks using Interval Merger Hyper or Hyperblock Rules Linear
     */
    public void generateHBs(boolean remove_old) {

        // moved variable declarations to here that need to be in scope of either version of the cuda algorithm.

        // Hyperblocks generated with this algorithm
        ArrayList<HyperBlock> gen_hb = new ArrayList<>();

        if (remove_old)
            hyper_blocks.clear();

        // get data to create hyperblocks
        ArrayList<ArrayList<DataATTR>> attributes = separateByAttribute(data);
        ArrayList<ArrayList<DataATTR>> all_intv = new ArrayList<>();

        // get number of threads (number of processors or number of attributes)
        int numThreads = Math.min(Runtime.getRuntime().availableProcessors(), attributes.size());
        ExecutorService executorService = Executors.newFixedThreadPool(numThreads);

        // Create dataset without data from interval Hyperblocks
        ArrayList<ArrayList<double[]>> datum = new ArrayList<>();
        ArrayList<ArrayList<double[]>> seed_data = new ArrayList<>();
        ArrayList<ArrayList<Integer>> skips = new ArrayList<>();

        // initially generate the blocks
        try {

            long startTime = System.currentTimeMillis();

            while (!attributes.get(0).isEmpty()) {
                // get largest hyperblock for each attribute
                ArrayList<DataATTR> intv = interval_hyper(executorService, attributes, acc_threshold, gen_hb);
                all_intv.add(intv);

                // if Hyperblock is unique then add
                if (intv.size() > 1) {
                    // Create and add new Hyperblock
                    ArrayList<ArrayList<double[]>> hb_data = new ArrayList<>();
                    ArrayList<double[]> intv_data = new ArrayList<>();

                    for (DataATTR dataATTR : intv)
                        intv_data.add(data.get(dataATTR.cl).data[dataATTR.cl_index]);

                    // Add data and Hyperblock
                    hb_data.add(intv_data);
                    HyperBlock hb = new HyperBlock(hb_data, intv.get(0).cl);
                    gen_hb.add(hb);
                }
                // break loop if a unique HB cannot be found
                else
                    break;
            }

            long endTime = System.currentTimeMillis();
            long executionTime = endTime - startTime;

            System.out.println("Parallelized Interval HB creation time: " + executionTime + " milliseconds");

            // add HBs and shutdown threads
            hyper_blocks.addAll(gen_hb);
            executorService.shutdown();

            // all data
            for (DataObject dataObject : data) {
                datum.add(new ArrayList<>(Arrays.asList(dataObject.data)));
                seed_data.add(new ArrayList<>());
                skips.add(new ArrayList<>());
            }

            // find which data to skip
            for (ArrayList<DataATTR> dataATTRS : all_intv) {
                for (DataATTR dataATTR : dataATTRS)
                    skips.get(dataATTR.cl).add(dataATTR.cl_index);
            }

            for (ArrayList<Integer> skip : skips)
                Collections.sort(skip);

            for (int i = 0; i < data.size(); i++) {
                for (int j = 0; j < data.get(i).data.length; j++) {
                    if (!skips.get(i).isEmpty()) {
                        if (j != skips.get(i).get(0))
                            seed_data.get(i).add(data.get(i).data[j]);
                        else
                            skips.get(i).remove(0);
                    } else
                        seed_data.get(i).add(data.get(i).data[j]);
                }
            }

            // sort data by most important attribute
            for (int i = 0; i < datum.size(); i++) {
                sortByColumn(datum.get(i), best_attribute);
                sortByColumn(seed_data.get(i), best_attribute);
            }
        }
        catch (InterruptedException | ExecutionException e) {
            LOGGER.log(Level.SEVERE, e.toString(), e);
            DV.warningPopup("Error Generating Hyperblocks", "Could not generate hyperblocks.");
            return;
        }

        // now if we made it through that, we can simply try the cuda version, if we get an exception for no cuda, then we use the not cuda version
        try {
            merger_cuda(datum, seed_data);
            order_hbs_by_class();
        }
        catch (InterruptedException | ExecutionException e) {
            LOGGER.log(Level.SEVERE, e.toString(), e);
            DV.warningPopup("Error Generating Hyperblocks", "Could not generate hyperblocks.");
            return;
        }
        // this catches the case where we don't have a cuda device, and runs it instead on the CPU
        catch(ExceptionInInitializerError | NoClassDefFoundError e) {

            // mini try catch that is exactly the same.
            try {
                // run dustin algorithm
                // create new threads equal to the number of processors
                executorService = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
                merger_hyperblock(executorService, datum, seed_data);
                executorService.shutdown();

                // order blocks from biggest to smallest by class
                order_hbs_by_class();
                // temp function removes all, but the largest HB in each class
                //order_hbs_by_size();
            } catch (InterruptedException | ExecutionException x) {
                LOGGER.log(Level.SEVERE, x.toString(), x);
                DV.warningPopup("Error Generating Hyperblocks", "Could not generate hyperblocks.");
                return;
            }
        }

        // SAVES HBS TO FILE FOR LATER USE
        // get the current date and time as a string
        /*String date = new SimpleDateFormat("yyyy-MM-dd_HH-mm-ss").format(new Date());
        String fileName = "D:\\Work\\Hyperblocks\\" + date + "_";

        System.out.println("Hyperblocks generated: " + hyper_blocks.size());
        saveHyperBlocksToCSV(fileName + "Hyperblocks.csv");
        saveDataObjectsToCSV(hyper_blocks.get(0).hyper_block, fileName + "HB1.csv");
        saveDataObjectsToCSV(hyper_blocks.get(1).hyper_block, fileName + "HB2.csv");*/
    }

    /***
     * Separates data into separate lists by attribute
     * @return Data separated by attribute
     */
    private ArrayList<ArrayList<DataATTR>> separateByAttribute(ArrayList<DataObject> data)
    {
        // data separated by attribute
        ArrayList<ArrayList<DataATTR>> attributes = new ArrayList<>();
        for (int k = 0; k < DV.fieldLength; k++)
        {
            ArrayList<DataATTR> tmpField = new ArrayList<>();
            for (int i = 0; i < data.size(); i++)
            {
                for (int j = 0; j < data.get(i).data.length; j++)
                    tmpField.add(new DataATTR(data.get(i).data[j][k], i, j));
            }

            // sort data by value then add
            tmpField.sort(Comparator.comparingDouble(o -> o.value));
            attributes.add(tmpField);
        }

        return attributes;
    }

    /**
     * Data Attribute. Stores one attribute of a datapoint and an identifying key shared with other attributes of the same datapoint.
     * @param value value of one attribute of a datapoint
     * @param cl class of a datapoint
     * @param cl_index index of point within class
     */
    private record DataATTR(double value, int cl, int cl_index) {}


    /**
     * Finds largest interval across all dimensions of a set of data.
     * @param executorService threads to run
     * @param data_by_attr all data split by attribute
     * @param acc_threshold accuracy threshold for interval
     * @param existing_hb existing hyperblocks to check for overlap
     * @return largest interval
     */
    private ArrayList<DataATTR> interval_hyper(ExecutorService executorService, ArrayList<ArrayList<DataATTR>> data_by_attr, double acc_threshold, ArrayList<HyperBlock> existing_hb) throws ExecutionException, InterruptedException
    {
        // future intervals
        List<Future<int[]>> intervals = new ArrayList<>();
        int attr = -1;
        int[] best = {-1, -1, -1};

        // search each attribute
        for (int i = 0; i < data_by_attr.size(); i++)
        {
            // get longest interval for attribute
            int finalI = i;
            Callable<int[]> task = () -> longest_interval(data_by_attr.get(finalI), acc_threshold, existing_hb, finalI);

            // execute task
            Future<int[]> future = executorService.submit(task);
            intervals.add(future);
        }

        // wait for results then find the largest interval
        for (Future<int[]> future : intervals)
        {
            int[] interval = future.get();
            if (interval[0] > 1 && interval[0] > best[0])
            {
                best[0] = interval[0];
                best[1] = interval[1];
                best[2] = interval[2];
                attr = interval[3];
            }
        }

        // construct ArrayList of data
        ArrayList<DataATTR> longest = new ArrayList<>();
        if (best[0] != -1)
        {
            for (int i = best[1]; i <= best[2]; i++)
                longest.add(data_by_attr.get(attr).get(i));
        }

        return longest;
    }


    /***
     * Finds the longest interval in a sorted list of data by attribute.
     * @param data_by_attr sorted data by attribute
     * @param acc_threshold accuracy threshold for interval
     * @param existing_hb existing hyperblocks to check for overlap
     * @param attr attribute to find interval on
     * @return longest interval
     */
    private int[] longest_interval(ArrayList<DataATTR> data_by_attr, double acc_threshold, ArrayList<HyperBlock> existing_hb, int attr)
    {
        // local and global intervals -> {size, interval start, interval end, attribute}
        int[] intr = new int[] {1, 0, 0};
        int[] max_intr = new int[] {-1, -1, -1, -1};

        // data size and misclassified data size
        int n = data_by_attr.size();
        double m = 0;

        for (int i = 1; i < n; i++)
        {
            // If current class matches with next
            if (data_by_attr.get(intr[1]).cl == data_by_attr.get(i).cl)
                intr[0]++;
            // purity is above threshold
            // ensure misclassified data is not classified correctly by LDF
            // classified correctly by LDF + misclassified by HB -> quit
            // misclassified by LDF + misclassified by HB -> continue
            else if (explain_ldf &&
                    misclassified_by_ldf(data.get(data_by_attr.get(i).cl).data[data_by_attr.get(i).cl_index]) &&
                    ((m + 1) / intr[0]) > acc_threshold)
            {
                m++;
                intr[0]++;
            }
            else if (!explain_ldf && ((m + 1) / intr[0]) > acc_threshold)
            {
                m++;
                intr[0]++;
            }
            else
            {
                // Remove value from interval if accuracy is below threshold
                if (data_by_attr.get(i - 1).value == data_by_attr.get(i).value)
                {
                    // remove then skip overlapped values
                    remove_value_from_interval(data_by_attr, intr, data_by_attr.get(i).value);
                    i = skip_value_in_interval(data_by_attr, i, data_by_attr.get(i).value);
                }

                // Update longest interval if it doesn't overlap
                if (intr[0] > max_intr[0] &&
                        check_interval_hyperblock_overlap(data_by_attr, intr, attr, existing_hb))
                {
                    max_intr[0] = intr[0];
                    max_intr[1] = intr[1];
                    max_intr[2] = intr[2];
                    max_intr[3] = attr;
                }

                // Reset current interval and misclassified points
                intr[0] = 1;
                intr[1] = i;
                m = 0;
            }

            // set end of interval to next datapoint
            intr[2] = i;
        }

        // Update longest interval if it doesn't overlap
        if (intr[0] > max_intr[0] &&
                check_interval_hyperblock_overlap(data_by_attr, intr, attr, existing_hb))
        {
            max_intr[0] = intr[0];
            max_intr[1] = intr[1];
            max_intr[2] = intr[2];
            max_intr[3] = attr;
        }

        // return largest interval
        return max_intr;
    }


    /**
     *
     * @param datapoint
     * @return
     */
    private boolean misclassified_by_ldf(double[] datapoint)
    {
        for (int i = 0; i < DV.misclassifiedData.size(); i++)
        {
            for (int j = 0; j < DV.misclassifiedData.get(i).size(); j++)
            {
                if (Arrays.equals(datapoint, DV.misclassifiedData.get(i).get(j)))
                    return true;
            }
        }

        return false;
    }

    
    /***
     *
     * @param low_bound
     * @param high_bound
     * @return
     */
    private ArrayList<ArrayList<double[]>> misclassified_by_ldf_in_range(double[] low_bound, double[] high_bound)
    {
        ArrayList<ArrayList<double[]>> in_range = new ArrayList<>();
        for (int i = 0; i < DV.misclassifiedData.size(); i++)
        {
            in_range.add(new ArrayList<>());

            for (int j = 0; j < DV.misclassifiedData.get(i).size(); j++)
            {
                boolean inside = true;

                for (int f = 0; f < DV.fieldLength; f++)
                {
                    if (DV.misclassifiedData.get(i).get(j)[f] > high_bound[f] || DV.misclassifiedData.get(i).get(j)[f] < low_bound[f])
                    {
                        inside = false;
                        break;
                    }
                }

                if (inside)
                    in_range.get(i).add(DV.misclassifiedData.get(i).get(j));
            }
        }

        return in_range;
    }


    /***
     *
     * @param data_by_attr
     * @param intr
     * @param value
     * @return
     */
    private void remove_value_from_interval(ArrayList<DataATTR> data_by_attr, int[] intr, double value)
    {
        while (data_by_attr.get(intr[2]).value == value)
        {
            if (intr[2] > intr[1])
            {
                intr[0]--;
                intr[2]--;
            }
            else
            {
                intr[0] = -1;
                break;
            }
        }
    }


    /***
     *
     * @param data_by_attr
     * @param index
     * @param value
     * @return
     */
    private int skip_value_in_interval(ArrayList<DataATTR> data_by_attr, int index, double value)
    {
        while (data_by_attr.get(index).value == value)
        {
            if (index < data_by_attr.size() - 1)
                index++;
            else
                break;
        }

        return index;
    }


    /**
     * Checks if a given interval overlaps with any existing hyperblock.
     * @param data_by_attr data interval exists on
     * @param intv interval to check
     * @param attr attribute interval exists on
     * @param existing_hb all existing hyperblocks
     * @return whether the interval is unique or not
     */

    private boolean check_interval_hyperblock_overlap(ArrayList<DataATTR> data_by_attr, int[] intv,  int attr, ArrayList<HyperBlock> existing_hb)
    {
        // get interval range
        double intv_min = data_by_attr.get(intv[1]).value;
        double intv_max = data_by_attr.get(intv[2]).value;

        // check if interval range overlaps with any existing hyperblocks
        // to not overlap the interval maximum must be below all existing hyperblock minimums
        // or the interval minimum must be above all existing hyperblock maximums
        for (HyperBlock hb : existing_hb)
        {
            // if not unique, then return false
            if (!(intv_max < hb.minimums.get(attr).get(0) || intv_min > hb.maximums.get(attr).get(0)))
            {
                return false;
            }
        }

        // if unique, then return true
        return true;
    }


    /***
     * Merger Hyperblock (MHyper) algorithm created by Boris Kovalerchuk and Dustin Hayes
     * Kovalerchuk B, Hayes D. Discovering Interpretable Machine Learning Models in Parallel Coordinates.
     * In 2021 25th International Conference Information Visualisation (IV) 2021 Jul 5 (pp. 181-188). IEEE, arXiv:2106.07474.
     * <a href="https://github.com/CWU-VKD-LAB/VisCanvas2.0">Code is translated from C++ from VisCanvas2.0</a>
     * @param executorService threads to run
     * @param data data to create hyperblocks from
     * @param out_data data not classified by interval hyperblocks
     */
    private void merger_hyperblock(ExecutorService executorService, ArrayList<ArrayList<double[]>> data, ArrayList<ArrayList<double[]>> out_data) throws ExecutionException, InterruptedException
    {
        // create hollow blocks from hyperblocks
        ArrayList<HollowBlock> merging_hbs = new ArrayList<>();
        for (HyperBlock hyperBlock : hyper_blocks)
        {
            double[] mins = new double[DV.fieldLength];
            double[] maxes = new double[DV.fieldLength];

            for(int i = 0; i < DV.fieldLength; i++){
                mins[i] = hyperBlock.minimums.get(i).get(0);
                maxes[i] = hyperBlock.maximums.get(i).get(0);
            }

            merging_hbs.add(new HollowBlock(Arrays.copyOf(maxes, maxes.length), Arrays.copyOf(mins, mins.length), hyperBlock.classNum));
        }

        hyper_blocks.clear();

        // create seed hollow blocks
        for (int i = 0; i < out_data.size(); i++)
        {
            for (int j = 0; j < out_data.get(i).size(); j++)
                merging_hbs.add(new HollowBlock(out_data.get(i).get(j), out_data.get(i).get(j), i));
        }

        boolean actionTaken = false;
        int cnt = merging_hbs.size();

        do
        {
            if (actionTaken || cnt < 1)
                cnt = merging_hbs.size();

            actionTaken = false;

            if (merging_hbs.isEmpty())
                break;

            int seedNum = -1;
            for (int i = 0; i < merging_hbs.size(); i++)
            {
                if (merging_hbs.get(i).mergable)
                {
                    seedNum = i;
                    break;
                }
            }

            if (seedNum == -1)
                break;

            HollowBlock seed_hb = merging_hbs.get(seedNum);
            merging_hbs.remove(seedNum);

            long startTime = System.currentTimeMillis();

            List<Future<Boolean>> futureActions = new ArrayList<>();
            ConcurrentLinkedQueue<double[][]> pointsToAdd = new ConcurrentLinkedQueue<>();
            AtomicIntegerArray toBeDeleted = new AtomicIntegerArray(merging_hbs.size());
            for (int i = 0; i < merging_hbs.size(); i++)
            {
                if (seed_hb.classNum == merging_hbs.get(i).classNum)
                {
                    // create callable to get merge result
                    int finalI = i;
                    Callable<Boolean> task = () -> merger_helper1(merging_hbs.get(finalI), seed_hb, finalI, data, toBeDeleted, pointsToAdd);

                    // execute task
                    Future<Boolean> future = executorService.submit(task);
                    futureActions.add(future);
                }
            }

            // wait for results then check if any action was taken
            for (Future<Boolean> future : futureActions)
            {
                if (future.get())
                    actionTaken = true;
            }

            long endTime = System.currentTimeMillis();
            long executionTime = endTime - startTime;



            // delete hyperblocks in reverse
            for (int i = toBeDeleted.length() - 1; i >= 0; i--)
            {
                if (toBeDeleted.get(i) == 1)
                    merging_hbs.remove(i);
            }

            // add successfully combined hyperblocks
            for (double[][] points : pointsToAdd)
            {
                HollowBlock hb = new HollowBlock(points[0], points[1], seed_hb.classNum);
                merging_hbs.add(hb);
            }

            if (!actionTaken)
            {
                seed_hb.mergable = false;
                merging_hbs.add(seed_hb);
            }

            System.out.println("Cnt: " + cnt);
            System.out.println("HB num: " + merging_hbs.size());
            System.out.println("Check time: " + executionTime + " milliseconds");

            cnt--;

        } while (actionTaken || cnt > 0);

        // reset mergable hollow blocks and find impure blocks
        cnt = merging_hbs.size();
        for (HollowBlock hollowBlock : merging_hbs)
        {
            if (!hollowBlock.mergable)
                hollowBlock.mergable = true;
        }

        do
         {
             if (actionTaken || cnt < 1)
                cnt = merging_hbs.size();

             actionTaken = false;

             if (merging_hbs.isEmpty())
                break;

             int seedNum = -1;
             for (int i = 0; i < merging_hbs.size(); i++)
             {
                 if (merging_hbs.get(i).mergable)
                 {
                     seedNum = i;
                     break;
                 }
             }

             if (seedNum == -1)
                 break;

            HollowBlock seed_hb = merging_hbs.get(seedNum);
            merging_hbs.remove(seedNum);

            long startTime = System.currentTimeMillis();

            ArrayList<Double> acc = new ArrayList<>();
            List<Future<Double>> futuresAccuracies = new ArrayList<>();
            ArrayList<Integer> toBeDeleted = new ArrayList<>();
            for (HollowBlock merging_hb : merging_hbs)
            {
                // create callable to get merge result
                Callable<Double> task = () -> merger_helper2(merging_hb, seed_hb, data);

                // execute task
                Future<Double> future = executorService.submit(task);
                futuresAccuracies.add(future);
            }

             // wait for results then check if any action was taken
             for (Future<Double> future : futuresAccuracies)
                 acc.add(future.get());

            int highestAccIndex = 0;
            for (int j = 0; j < acc.size(); j++)
            {
                if (acc.get(j) > acc.get(highestAccIndex))
                    highestAccIndex = j;
            }

            // if acc meets threshold
            if (acc.get(highestAccIndex) >= acc_threshold)
            {
                actionTaken = true;

                double[] maxPoint = new double[DV.fieldLength];
                double[] minPoint = new double[DV.fieldLength];

                // define combined space
                for (int j = 0; j < DV.fieldLength; j++)
                {
                    double newLocalMax = Math.max(seed_hb.maximums[j], merging_hbs.get(highestAccIndex).maximums[j]);
                    double newLocalMin = Math.min(seed_hb.minimums[j], merging_hbs.get(highestAccIndex).minimums[j]);

                    maxPoint[j] = newLocalMax;
                    minPoint[j] = newLocalMin;
                }

                int totalPointsInSpace = 0;
                int pointsInSeedSpace = 0;
                for (int i = 0; i < data.size(); i++)
                {
                    for (double[] doubles : data.get(i))
                    {
                        boolean withinSpace = true;
                        double[] tmp_pnt = new double[DV.fieldLength];

                        for (int w = 0; w < DV.fieldLength; w++)
                        {
                            tmp_pnt[w] = doubles[w];
                            if (!(tmp_pnt[w] <= maxPoint[w] && tmp_pnt[w] >= minPoint[w]))
                            {
                                withinSpace = false;
                                break;
                            }
                        }

                        if (withinSpace)
                        {
                            totalPointsInSpace++;
                            if (seed_hb.classNum == i)
                                pointsInSeedSpace++;
                        }
                    }
                }

                int cls = (double) pointsInSeedSpace / totalPointsInSpace > 0.5 ? seed_hb.classNum : merging_hbs.get(highestAccIndex).classNum;
                HollowBlock hb = new HollowBlock(maxPoint, minPoint, cls);
                merging_hbs.add(hb);

                // store this index to delete the cube that was combined
                toBeDeleted.add(highestAccIndex);
            }

             long endTime = System.currentTimeMillis();
             long executionTime = endTime - startTime;

            toBeDeleted.sort(Collections.reverseOrder());
            for (int i = 0; i < toBeDeleted.size(); i++)
                merging_hbs.remove(i);

             if (!actionTaken)
                 seed_hb.mergable = false;

            merging_hbs.add(seed_hb);

             System.out.println("2nd Cnt: " + cnt);
             System.out.println("2nd HB num: " + merging_hbs.size());
             System.out.println("2nd Check time: " + executionTime + " milliseconds");

            cnt--;

            } while (actionTaken || cnt > 0);

            // create hyperblocks from mergine blocks
            for (HollowBlock mergingHb : merging_hbs) {
                ArrayList<ArrayList<Double>> mins = new ArrayList<>();
                ArrayList<ArrayList<Double>> maxes = new ArrayList<>();

                for (int i = 0; i < mergingHb.minimums.length; i++) {
                    // Convert each minimum and maximum into an ArrayList<Double>
                    ArrayList<Double> minList = new ArrayList<>();
                    ArrayList<Double> maxList = new ArrayList<>();

                    minList.add(mergingHb.minimums[i]);
                    maxList.add(mergingHb.maximums[i]);

                    mins.add(minList);
                    maxes.add(maxList);
                }

                // Add them to the list of blocks
                hyper_blocks.add(new HyperBlock(new ArrayList<>(maxes), new ArrayList<>(mins), mergingHb.classNum));
            }
    }


    public static double[][] findMaxDistance(ConcurrentLinkedQueue<double[][]> boundsList) {
        double maxDistance = -1;
        double[][] maxDistanceBounds = new double[3][];

        int cnt = 0;
        for (double[][] bounds : boundsList) {
            double[] upperBounds = bounds[0];
            double[] lowerBounds = bounds[1];
            double distance = calculateSpace(lowerBounds, upperBounds);

            if (distance > maxDistance) {
                maxDistance = distance;
                maxDistanceBounds[0] = bounds[0];
                maxDistanceBounds[1] = bounds[1];
                maxDistanceBounds[2] = new double[]{cnt};
            }

            cnt++;
        }

        return maxDistanceBounds;
    }


    private static double calculateSpace(double[] lowerBounds, double[] upperBounds) {
        double sum = 0;
        for (int i = 0; i < lowerBounds.length; i++) {
            double diff = upperBounds[i] - lowerBounds[i];
            sum += diff;
        }
        return sum;
    }


    private int[] get_interval_start(ArrayList<ArrayList<double[]>> data, double min)
    {
        int[] start = new int[data.size()];
        for (int i = 0; i < data.size(); i++)
        {
            for (int j = 0; j < data.get(i).size(); j++)
            {
                if (data.get(i).get(j)[best_attribute] >= min)
                {
                    start[i] = j;
                    break;
                }
            }
        }

        return start;
    }


    private int[] get_interval_end(ArrayList<ArrayList<double[]>> data, double max)
    {
        int[] end = new int[data.size()];
        for (int i = 0; i < data.size(); i++)
        {
            for (int j = data.get(i).size() - 1; j >= 0; j--)
            {
                if (data.get(i).get(j)[best_attribute] <= max)
                {
                    end[i] = j;
                    break;
                }
            }
        }

        return end;
    }


    private boolean merger_helper1(HollowBlock merging_hb, HollowBlock seed_hb, int i, ArrayList<ArrayList<double[]>> data, AtomicIntegerArray toBeDeleted, ConcurrentLinkedQueue<double[][]> pointsToAdd)
    {
        double[] maxPoint = new double[DV.fieldLength];
        double[] minPoint = new double[DV.fieldLength];

        // define combined space
        for (int j = 0; j < DV.fieldLength; j++)
        {
            double newLocalMax = Math.max(seed_hb.maximums[j], merging_hb.maximums[j]);
            double newLocalMin = Math.min(seed_hb.minimums[j], merging_hb.minimums[j]);

            maxPoint[j] = newLocalMax;
            minPoint[j] = newLocalMin;
        }

        // get start and end of interval
        int[] start = get_interval_start(data, minPoint[best_attribute]);//new int[]{0, 0};
        int[] end = get_interval_end(data, maxPoint[best_attribute]);//new int[]{data.get(0).size()-1, data.get(1).size()-1};

        //System.out.println("Start: " + Arrays.toString(start));
        //System.out.println("End: " + Arrays.toString(end));

        ArrayList<Integer> classInSpace = new ArrayList<>();
        for (int j = 0; j < data.size(); j++)
        {
            for (int k = start[j]; k < end[j]; k++)
            {
                boolean withinSpace = true;
                double[] tmp_pnt = new double[DV.fieldLength];
                for (int w = 0; w < DV.fieldLength; w++)
                {
                    tmp_pnt[w] = data.get(j).get(k)[w];
                    if (!(tmp_pnt[w] <= maxPoint[w] && tmp_pnt[w] >= minPoint[w]))
                    {
                        withinSpace = false;
                        break;
                    }
                }

                if (withinSpace)
                    classInSpace.add(j);
            }
        }

        // check if new space is pure
        HashSet<Integer> classCnt = new HashSet<>(classInSpace);
        if (classCnt.size() <= 1)
        {
            pointsToAdd.add(new double[][]{maxPoint, minPoint});
            toBeDeleted.set(i, 1);
            return true;
        }
        else
            return false;
    }


    private double merger_helper2(HollowBlock merging_hb, HollowBlock seed_hb, ArrayList<ArrayList<double[]>> data)
    {
        double[] maxPoint = new double[DV.fieldLength];
        double[] minPoint = new double[DV.fieldLength];

        // define combined space
        for (int j = 0; j < DV.fieldLength; j++)
        {
            double newLocalMax = Math.max(seed_hb.maximums[j], merging_hb.maximums[j]);
            double newLocalMin = Math.min(seed_hb.minimums[j], merging_hb.minimums[j]);

            maxPoint[j] = newLocalMax;
            minPoint[j] = newLocalMin;
        }

        ArrayList<Integer> classInSpace = new ArrayList<>();
        ArrayList<ArrayList<double[]>> pointsInSpace = new ArrayList<>();

        for (int j = 0; j < data.size(); j++)
        {
            pointsInSpace.add(new ArrayList<>());
            for (int k = 0; k < data.get(j).size(); k++)
            {
                boolean withinSpace = true;
                double[] tmp_pnt = new double[DV.fieldLength];

                for (int w = 0; w < DV.fieldLength; w++)
                {
                    tmp_pnt[w] = data.get(j).get(k)[w];

                    if (!(tmp_pnt[w] <= maxPoint[w] && tmp_pnt[w] >= minPoint[w]))
                    {
                        withinSpace = false;
                        break;
                    }
                }

                if (withinSpace)
                {
                    classInSpace.add(j);
                    pointsInSpace.get(j).add(tmp_pnt);
                }
            }
        }

        if (!classInSpace.isEmpty())
        {
            // check if new space is pure enough
            HashMap<Integer, Integer> classCnt = new HashMap<>();
            for (int ints : classInSpace)
            {
                if (classCnt.containsKey(ints))
                    classCnt.replace(ints, classCnt.get(ints) + 1);
                else
                    classCnt.put(ints, 1);
            }

            int majorityClass = classInSpace.get(0);
            for (int key : classCnt.keySet())
            {
                if (classCnt.get(key) > classCnt.get(majorityClass))
                    majorityClass = key;
            }

            double curClassTotal = 0;
            double classTotal = 0;

            for (int key : classCnt.keySet())
            {
                if (key == majorityClass)
                    curClassTotal = classCnt.get(key);

                classTotal += classCnt.get(key);
            }

            double cur_acc = curClassTotal / classTotal;

            // check if a point misclassified by the LDF lies in the new space
            if (explain_ldf)
            {
                // collect misclassified data within range
                ArrayList<ArrayList<double[]>> mis_in_range = misclassified_by_ldf_in_range(minPoint, maxPoint);

                // ensure that data classified correctly by LDF is also correct in HB
                // only check data not in majority (not classified correctly)
                for (int i = 0; i < pointsInSpace.size(); i++)
                {
                    if (i != majorityClass)
                    {
                        for (int j = 0; j < pointsInSpace.get(i).size(); j++)
                        {
                            boolean misclassified_by_both = false;

                            for (int k = 0; k < mis_in_range.get(i).size(); k++)
                            {
                                if (Arrays.equals(pointsInSpace.get(i).get(j), mis_in_range.get(i).get(k)))
                                {
                                    misclassified_by_both = true;
                                    break;
                                }
                            }

                            if (!misclassified_by_both)
                            {
                                cur_acc = -1;
                                break;
                            }
                        }
                    }

                    if (cur_acc == -1)
                        break;
                }
            }

            return cur_acc;
        }
        else
            return -1;
    }


    static class HBComparator implements Comparator<HyperBlock>
    {
        // Gives largest Hyperblock
        public int compare(HyperBlock b1, HyperBlock b2)
        {
            int b1_size = 0;
            int b2_size = 0;

            for (int i = 0; i < b1.hyper_block.size(); i++)
                b1_size += b1.hyper_block.get(i).size();

            for (int i = 0; i < b2.hyper_block.size(); i++)
                b2_size += b2.hyper_block.get(i).size();

            return Integer.compare(b2_size, b1_size);
        }
    }


    /**
     * Generalizes Hyperblocks
     */
    public void increase_level()
    {
        data.clear();

        // create dataset
        ArrayList<ArrayList<double[]>> tmpData = new ArrayList<>();
        for (int i = 0; i < DV.classNumber; i++)
            tmpData.add(new ArrayList<>());

        // Create a HashSet to track unique data points
        HashSet<String> uniqueDataPoints = new HashSet<>();
        for (int i = 0; i < hyper_blocks.size(); i++)
        {
            ArrayList<double[]> envelope = find_envelope_cases(i);
            for (double[] doubles : envelope)
            {
                // Convert the data point to a string to check for uniqueness
                String dataPointStr = Arrays.toString(doubles);
                if (!uniqueDataPoints.contains(dataPointStr))
                {
                    uniqueDataPoints.add(dataPointStr);
                    tmpData.get(hyper_blocks.get(i).classNum).add(doubles);
                }
            }
        }

        misclassified.clear();
        accuracy.clear();

        for (int i = 0; i < tmpData.size(); i++)
        {
            double[][] newData = new double[tmpData.get(i).size()][DV.fieldLength];
            tmpData.get(i).toArray(newData);
            DataObject newObj = new DataObject(DV.trainData.get(i).className, newData);
            newObj.updateCoordinatesGLC(DV.angles);
            data.add(newObj);
        }

        generateHBs(true);
        HB_analytics();

        // visualize hyperblocks
        if (!hyper_blocks.isEmpty())
            getPlot();

        System.out.println("INCREASE LEVEL COMPLETE");
    }

    private void getOverlapData()
    {
        data = new ArrayList<>();

        // create DataObjects of overlapping points
        ArrayList<DataObject> noOverlap = new ArrayList<>(List.of(
                new DataObject("upper", Analytics.convertArrayListToDoubleArray(Analytics.upper)),
                new DataObject("lower", Analytics.convertArrayListToDoubleArray(Analytics.lower))
        ));

        data.addAll(noOverlap);
    }

    private void getNonOverlapData()
    {

    }

    private void getData()
    {
        // get classes to be graphed
        data = new ArrayList<>();
        data.addAll(DV.trainData);
    }
    
    
    /**
     *                          *
     * ------------------------ *
     * VISUALIZATION FUNCTIONS  *
     * ------------------------ *
     *                          *
     */

    /**
     * Create GUI for hyperblock visualization
     */
    private void HB_GUI()
    {
        JFrame mainFrame = new JFrame();
        mainFrame.setLayout(new BorderLayout());

        // create toolbar
        mainFrame.add(createToolBar(), BorderLayout.PAGE_START);

        // create graph panel
        mainFrame.add(createGraphPanel(), BorderLayout.CENTER);

        // create navigation panel
        createNavPanel();
        mainFrame.add(navPanel, BorderLayout.PAGE_END);

        // Make the interactive window
        createExpansionPanel();
        expansionScroll.setVisible(false);
        mainFrame.add(expansionScroll, BorderLayout.EAST);
        mainFrame.setMinimumSize(new Dimension(DV.minSize[0], DV.minSize[1]));

        // show
        mainFrame.setVisible(true);
        mainFrame.revalidate();
        mainFrame.pack();
        mainFrame.repaint();
    }

    private void createExpansionPanel() {
        expansionPanel = new JPanel();
        expansionPanel.setPreferredSize(new Dimension(200, 1080));
        expansionPanel.setBorder(BorderFactory.createTitledBorder("Interactive Interval Expansion."));
        expansionPanel.setVisible(true);

        expansionPanel.add(new JLabel("Expand Attributes"));
        boolean[] checkBoxValues = new boolean[DV.fieldLength];
        for(int i = 0; i < DV.fieldLength; i++){
            // Make new checkbox for each attribute
            JCheckBox attributeBox = new JCheckBox("x" + i + " : " + DV.fieldNames.get(i));

            // Default all to true, since more likely.
            attributeBox.setSelected(true);
            checkBoxValues[i] = true;

            final int index = i;
            attributeBox.addChangeListener(e ->{
                checkBoxValues[index] = attributeBox.isSelected();
            });
            expansionPanel.add(attributeBox);
        }

        // Allow user to choose which attributes to expand

        expansionPanel.add(new JLabel("Expand by: "));

        // Allow user to enter number between 0.0 and 1.0 to attempt to expand by
        SpinnerNumberModel model = new SpinnerNumberModel(0.05, 0.0, 1.0, 0.01);
        JSpinner spinner = new JSpinner(model);
        JSpinner.NumberEditor editor = new JSpinner.NumberEditor(spinner, "0.00");
        spinner.setEditor(editor);
        expansionPanel.add(spinner);

        JCheckBox keepRealEdge = new JCheckBox("Keep edges as real point values.");
        keepRealEdge.setToolTipText("Tries to expand mins/maxes to the closest real point value that increases range. Ensures interval edges are true cases.");
        expansionPanel.add(keepRealEdge);

        // Change current block or change current block option
        JButton expandCurrent = new JButton("Expand Current Block");
        expandCurrent.addActionListener(e->{
            System.out.println("Expanding block " + visualized_block);
            attemptToExpandIntervals((double) spinner.getValue(), visualized_block, keepRealEdge.isSelected(), checkBoxValues);
            updateGraphs();
            HB_analytics();
        });

        // The button to expand all the blocks at the same time by specified amount
        JButton expandAll = new JButton("Expand All Blocks");
        expandAll.addActionListener(e->{
            System.out.println("Expanding all blocks.");
            attemptToExpandIntervals((double) spinner.getValue(), -1, keepRealEdge.isSelected(), checkBoxValues);
            updateGraphs();
            HB_analytics();
        });


        expansionPanel.add(expandCurrent);
        expansionPanel.add(expandAll);

        JButton closeButton = new JButton("Close");
        closeButton.addActionListener(e->{
            expansionScroll.setVisible(false);
        });

        // The button to hide the interactive panel.
        expansionPanel.add(closeButton);


        expansionScroll = new JScrollPane(expansionPanel);
        expansionScroll.setPreferredSize(new Dimension(220, 600));
        expansionScroll.setVerticalScrollBarPolicy(JScrollPane.VERTICAL_SCROLLBAR_ALWAYS);
    }

    /**
     * Create a toolbar for the hyperblock GUI
     * @return toolbar
     */
    private JToolBar createToolBar() {
        JToolBar toolBar = new JToolBar();
        toolBar.setFloatable(false);
        toolBar.setRollover(true);

        // Dropdown for plot options
        JLabel plots = new JLabel("Plot Options: ");
        plots.setFont(plots.getFont().deriveFont(Font.BOLD, 12f));
        toolBar.add(plots);
        toolBar.addSeparator();

        plotOptions = new JComboBox<>(new String[] {"PC", "GLC-L", "SPC-1B", "PC-2n", "SPC-2B"});
        plotOptions.addActionListener(e -> {
            String selected = (String) plotOptions.getSelectedItem();
            switch (selected) {
                case "PC": plot_id = 0; break;
                case "GLC-L": plot_id = 1; break;
                case "SPC-1B": plot_id = 2; break;
                case "PC-2n": plot_id = 3; break;
                case "SPC-2B": plot_id = 4; break;
            }
            updateGraphs();
        });
        toolBar.add(plotOptions);
        toolBar.addSeparator();

        // Dropdown for view options
        JLabel views = new JLabel("View Options: ");
        views.setFont(views.getFont().deriveFont(Font.BOLD, 12f));
        toolBar.add(views);
        toolBar.addSeparator();

        viewOptions = new JComboBox<>(new String[] {
                "Individual Hyperblocks", "All Blocks", "Class Combined Blocks", "Combined Blocks"
        });
        viewOptions.addActionListener(e -> {
            getPlot();
        });
        toolBar.add(viewOptions);
        toolBar.addSeparator();

        // Dropdown for hyperblock options
        JLabel blockType = new JLabel("Hyperblock Options: ");
        blockType.setFont(blockType.getFont().deriveFont(Font.BOLD, 12f));
        toolBar.add(blockType);
        toolBar.addSeparator();

        JComboBox<String> blockOptions = new JComboBox<>(new String[] {
                "Default Blocks", "Overlap Blocks", "Non-Overlap Blocks"
        });
        blockOptions.addActionListener(e -> {
            String selected = (String) blockOptions.getSelectedItem();
            if (selected.equals("Default Blocks")) getData();
            else if (selected.equals("Overlap Blocks")) getOverlapData();
            else if (selected.equals("Non-Overlap Blocks")) getNonOverlapData();
            updateGraphs();
        });
        toolBar.add(blockOptions);
        toolBar.addSeparator();

        // Dropdown for data view options
        JLabel dataView = new JLabel("Data View Options: ");
        dataView.setFont(dataView.getFont().deriveFont(Font.BOLD, 12f));
        toolBar.add(dataView);
        toolBar.addSeparator();

        dataViewOptions = new JComboBox<>(new String[] {
                "Within Block", "All Data", "Outline Block"
        });
        dataViewOptions.addActionListener(e -> updateGraphs());
        toolBar.add(dataViewOptions);
        toolBar.addSeparator();

        // Simplification options
        JLabel simplificationsL = new JLabel("View Options: ");
        simplificationsL.setFont(simplificationsL.getFont().deriveFont(Font.BOLD, 12f));
        toolBar.add(simplificationsL);
        toolBar.addSeparator();

        simplifications = new JComboBox<>(new String[] {
                "Remove Useless Attributes",
                "Create Disjunctive Blocks",
                "Remove Useless Blocks",
                "Expansion Algorithm" //Currently disabled until made interactive and ran by Dr. K
        });
        simplifications.addActionListener(e ->{
            String selected = (String) simplifications.getSelectedItem();
            if (selected.equals("Remove Useless Attributes")) removeUselessAttributes();
            else if (selected.equals("Create Disjunctive Blocks")) simplifyHBtoDisjunctiveForm();
            else if (selected.equals("Remove Useless Blocks")) removeUselessBlocks();
            else if (selected.equals("Expansion Algorithm")){
                expansionScroll.setVisible(true);
                //attemptToExpandIntervals(.05, -1, false);
            }

            // Add that the algo has run to the log.
            simplificationAlgoLog.add(selected);

            // Create statistics for the blocks resulting from the simplification ran
            blockStats.updateHyperBlockStatistics();

            HB_analytics();
            updateGraphs();
        });

        toolBar.add(simplifications);
        toolBar.addSeparator();

        JButton statistics = new JButton("Block Statistics");
        toolBar.add(statistics);
        //TODO:AUSTIN: THIS SHOULD NOT MAKE A NEW ONE, INSTEAD SHOULD SHOW THE WINDOW OF EXISTING OBJECT TO THE USER.
        statistics.addActionListener(e -> {
            blockStats.consoleStatistics();
        });

        JLabel lvlView = new JLabel("HB Level: ");
        lvlView.setFont(lvlView.getFont().deriveFont(Font.BOLD, 12f));
        toolBar.add(lvlView);

        hb_lvl = new JSpinner(new SpinnerNumberModel(1, 1, 3, 1));
        hb_lvl.setEditor(new JSpinner.NumberEditor(hb_lvl, "0"));
        hb_lvl.addChangeListener(hbe -> {
            if ((Integer) hb_lvl.getValue() == 2 || (Integer) hb_lvl.getValue() == 3) {
                increase_level();
                updateGraphs();
            }
        });
        toolBar.add(hb_lvl);

        return toolBar;
    }


    /**
     * Create a navigation panel for the hyperblock GUI
     */
    private void createNavPanel()
    {
        // set initial description
        graphLabel.setText(HB_desc(visualized_block));

        // left and right hyperblock navigation buttons
        left = new JButton("Previous Block");
        left.addActionListener(e ->
        {
            visualized_block--;
            if (visualized_block < 0)
                visualized_block = hyper_blocks.size() - 1;

            graphLabel.setText(HB_desc(visualized_block));
            updateGraphs();
        });

        right = new JButton("Next Block");
        right.addActionListener(e ->
        {
            visualized_block++;
            if (visualized_block > hyper_blocks.size() - 1)
                visualized_block = 0;

            graphLabel.setText(HB_desc(visualized_block));
            updateGraphs();
        });

        navPanel = new JPanel();
        navPanel.setLayout(new GridLayout(1, 2));
        navPanel.add(left);
        navPanel.add(right);
    }


    /**
     * Create a graph panel with a description for the hyperblock GUI
     * @return graph panel and graph label panel
     */
    private JPanel createGraphPanel()
    {
        // visualization panel and label
        graphPanel = new JPanel();
        graphPanel.setLayout(new BoxLayout(graphPanel, BoxLayout.PAGE_AXIS));

        graphLabel = new JLabel("");
        graphLabel.setFont(graphLabel.getFont().deriveFont(20f));

        // create panel with graph and graph label
        JPanel panel = new JPanel();
        panel.setLayout(new GridBagLayout());

        GridBagConstraints c = new GridBagConstraints();
        c.weightx = 1;
        c.weighty = 1;
        c.fill = GridBagConstraints.BOTH;
        c.ipady = 10;
        c.gridx = 0;
        c.gridy = 0;
        c.gridwidth = 2;
        panel.add(graphPanel, c);

        c.gridy = 1;
        c.weighty = 0;
        panel.add(graphLabel, c);

        // get colors
        getGraphColors();

        return panel;
    }


    /**
     * Get colors for the hyperblock visualization
     */
    private void getGraphColors()
    {
        graphColors = new Color[DV.classNumber];
        for (int i = 0; i < DV.classNumber; i++)
        {
            if (i == 0)
                graphColors[i] = Color.GREEN;
            else if (i == 1)
                graphColors[i] = Color.RED;
            else if (i == 2)
                graphColors[i] = Color.BLUE;
            else if (i == 3)
                graphColors[i] = Color.MAGENTA;
            else if (i == 4)
                graphColors[i] = Color.BLACK;
            else if (i == 5)
                graphColors[i] = Color.GRAY;
            else if (i == 6)
                graphColors[i] = Color.CYAN;
            else if (i == 7)
                graphColors[i] = Color.PINK;
            else
            {
                float r = (float) Math.random();
                float g = (float) Math.random();
                float b = (float) Math.random();
                graphColors[i] = new Color(r, g, b);
            }
        }
    }


    /**
     * Selects plot to visualize hyperblocks
     */
    private void getPlot()
    {
        graphPanel.removeAll();

        // Get the selected plot type from the plot options dropdown
        String selectedPlot = (String) plotOptions.getSelectedItem();
        String selectedView = (String) viewOptions.getSelectedItem();

        // Determine the visualization based on dropdown selections
        switch (selectedPlot)
        {
            case "PC" -> {
                switch (selectedView) {
                    case "Individual Hyperblocks" -> graphPanel.add(PC_HB(data, visualized_block));
                    case "All Blocks" -> graphPanel.add(PC_HBs(data));
                    case "Class Combined Blocks" -> {
                        // Add your logic for class combined blocks here
                    }
                    case "Combined Blocks" -> {
                        // Add your logic for combined blocks here
                    }
                }
            }
            case "GLC-L" -> {
                switch (selectedView) {
                    case "Individual Hyperblocks" -> graphPanel.add(GLC_L_HB(data, visualized_block));
                    case "All Blocks" -> {
                        // Add your logic for all blocks here
                    }
                    case "Class Combined Blocks" -> {
                        // Add your logic for class combined blocks here
                    }
                    case "Combined Blocks" -> {
                        // Add your logic for combined blocks here
                    }
                }
            }
            case "SPC-1B" -> {
                switch (selectedView) {
                    case "Individual Hyperblocks" -> {
                        // Add your logic for individual hyperblocks here
                    }
                    case "All Blocks" -> {
                        // Add your logic for all blocks here
                    }
                    case "Class Combined Blocks" -> {
                        // Add your logic for class combined blocks here
                    }
                    case "Combined Blocks" -> {
                        // Add your logic for combined blocks here
                    }
                }
            }
            case "PC-2n" -> {
                switch (selectedView) {
                    case "Individual Hyperblocks" -> {
                        // Add your logic for individual hyperblocks here
                    }
                    case "All Blocks" -> {
                        // Add your logic for all blocks here
                    }
                    case "Class Combined Blocks" -> {
                        // Add your logic for class combined blocks here
                    }
                    case "Combined Blocks" -> {
                        // Add your logic for combined blocks here
                    }
                }
            }
            case "SPC-2B" -> {
                switch (selectedView) {
                    case "Individual Hyperblocks" -> {
                        // Add your logic for individual hyperblocks here
                    }
                    case "All Blocks" -> {
                        // Add your logic for all blocks here
                    }
                    case "Class Combined Blocks" -> {
                        // Add your logic for class combined blocks here
                    }
                    case "Combined Blocks" -> {
                        // Add your logic for combined blocks here
                    }
                }
            }
            default -> graphPanel.add(PC_HB(data, visualized_block));
        }

        // Refresh the visualization
        graphPanel.revalidate();
        graphPanel.repaint();
    }
    
    /**
     * Prints the rules for the hyper-block.
     * @param block The int index of the hyper-block in the list hyper_blocks
     * @return
     */
    private String HB_desc(int block)
    {
        StringBuilder rule = new StringBuilder("<b>Rule:</b> if ");
        HyperBlock tempB = hyper_blocks.get(block);

        // Only print rules if it is not too long
        if (DV.fieldLength < 50)
        {
            // Go through all the attributes.
            for (int i = 0; i < DV.fieldLength; i++)
            {
                if(tempB.maximums.get(i).get(0) == 1 && tempB.minimums.get(i).get(0) == 0){
                    if(i == DV.fieldLength - 1){
                        rule.append(" then class ").append(DV.uniqueClasses.get(tempB.classNum));
                    }

                    continue;
                }

                // Go through all allowed intervals for the attribute
                for(int j = 0; j < tempB.maximums.get(i).size(); j++){
                    if(tempB.maximums.get(i).get(j) > 0){
                        // If multiple intervals appear
                        if(j > 0){
                            rule.append(" OR ");
                        }

                        rule.append(String.format("%.2f &le; X%d &le; %.2f", tempB.minimums.get(i).get(j), i, tempB.maximums.get(i).get(j)));
                    }
                }

                if (i != DV.fieldLength - 1)
                    rule.append(", ");
                else
                    rule.append(", then class ").append(DV.uniqueClasses.get(tempB.classNum));
            }
        }
        else
        {
            rule = new StringBuilder("<b>Rule:</b> N/A");
        }

        String desc = "<html><b>Block:</b> " + (block + 1) + "/" + hyper_blocks.size() + "<br/>";
        desc += "<b>Class:</b> " + DV.uniqueClasses.get(tempB.classNum) + "<br/>";
        desc += "<b>Datapoints:</b> " + tempB.size + " (" + misclassified.get(block) + " misclassified)" + "<br/>";
        desc += "<b>Accuracy:</b> " + (Math.round(accuracy.get(block) * 10000) / 100.0) + "%<br/>";
        desc += rule;
        desc += "</html>";

        return desc;
    }

    private void updateGraphs() {

        /*

        String selectedOption = (String) viewOptions.getSelectedItem();
        switch (selectedOption) {
            case "Individual Hyperblocks" -> {
                // Equivalent to individualView.doClick();
                individualViewAction();
            }
            case "All Blocks" -> {
                // Equivalent to completeView.doClick();
                completeViewAction();
            }
            case "Class Combined Blocks" -> {
                // Equivalent to combinedClassView.doClick();
                combinedClassViewAction();
            }
            default -> {
                // Default case for Combined View
                combinedViewAction();
            }
        }
        */
        // set left and right buttons
        navPanel.removeAll();
        navPanel.add(left);
        navPanel.add(right);

        if (visualized_block > hyper_blocks.size() - 1)
            visualized_block = 0;

        getPlot();
    }


    /**
     * Create a PC visualization for a single hyper-block
     * @param datum data to visualize
     * @param visualized_block hyper-block to visualize
     * @return panel with PC visualization
     */
    private ChartPanel PC_HB(ArrayList<DataObject> datum, int visualized_block)
    {
        HyperBlock tempBlock = hyper_blocks.get(visualized_block);

        // Create main renderers and datasets
        XYLineAndShapeRenderer goodLineRenderer = new XYLineAndShapeRenderer(true, false);
        XYSeriesCollection goodGraphLines = new XYSeriesCollection();
        XYLineAndShapeRenderer badLineRenderer = new XYLineAndShapeRenderer(true, false);
        XYSeriesCollection badGraphLines = new XYSeriesCollection();

        // Average renderer and dataset
        XYLineAndShapeRenderer avgRenderer = new XYLineAndShapeRenderer(true, false);
        XYSeriesCollection avgLines = new XYSeriesCollection();


        // renders the hyperblock outline and highlights the area
        XYLineAndShapeRenderer pcBlockRenderer = new XYLineAndShapeRenderer(true, false);
        // The area part.
        XYAreaRenderer pcBlockAreaRenderer = new XYAreaRenderer(XYAreaRenderer.AREA);

        BasicStroke[] strokes = createStrokes(visualized_block);

        int lineCnt = 0;
        // Loops through the data objects
        for (int d = 0; d < datum.size(); d++)
        {
            // Go through the datapoints, add to
            for (int i = 0; i < datum.get(d).data.length; i++)
            {
                // Start line at (0, 0)
                XYSeries line = new XYSeries(lineCnt, false, true);

                boolean within = inside_HB(visualized_block, datum.get(d).data[i]);

                // Add points to lines
                int off = 0;
                // Goes through all attributes of the same point
                for (int j = 0; j < DV.fieldLength; j++)
                {
                    if (remove_extra && tempBlock.minimums.get(j).get(0) != 0.5 && tempBlock.maximums.get(j).get(0) != 0.5)
                    {
                        line.add(off, datum.get(d).data[i][j]);
                        off++;
                    }
                    else if (!remove_extra)
                    {
                        line.add(j, datum.get(d).data[i][j]);
                    }
                }

                // Add series based on selection in dataViewOptions
                String selectedOption = (String) dataViewOptions.getSelectedItem();
                if ("Within Block".equals(selectedOption) && within)
                {
                    // add series if of same class

                    if (d == tempBlock.classNum)
                    {
                        goodGraphLines.addSeries(line);
                        goodLineRenderer.setSeriesPaint(lineCnt, graphColors[d]);
                        goodLineRenderer.setSeriesStroke(lineCnt, strokes[0]);
                    }
                    else
                    {
                        badGraphLines.addSeries(line);
                        badLineRenderer.setSeriesPaint(lineCnt, graphColors[d]);
                        badLineRenderer.setSeriesStroke(lineCnt, strokes[0]);
                    }
                    lineCnt++;
                }
                else if ("All Data".equals(selectedOption))
                {
                    goodGraphLines.addSeries(line);
                    goodLineRenderer.setSeriesPaint(lineCnt, graphColors[d]);
                    goodLineRenderer.setSeriesStroke(lineCnt, strokes[0]);
                    lineCnt++;
                }
                else if ("Outline Block".equals(selectedOption))
                {
                    // Handle "Outline Block" logic here if needed
                }
            }
        }

        // Populate average series
        for (int k = 0; k < average_case.get(visualized_block).size(); k++)
        {
            if (tempBlock.hyper_block.get(k).size() > 1)
            {
                XYSeries line = new XYSeries(k, false, true);
                int off = 0;
                for (int i = 0; i < average_case.get(visualized_block).get(k).length; i++)
                {
                    if (remove_extra && tempBlock.minimums.get(i).get(0) != 0.5 && tempBlock.maximums.get(i).get(0) != 0.5)
                    {
                        line.add(off, average_case.get(visualized_block).get(k)[i]);
                        off++;
                    }
                    else if (!remove_extra)
                    {
                        line.add(i, average_case.get(visualized_block).get(k)[i]);
                    }
                }

                avgLines.addSeries(line);
                avgRenderer.setSeriesPaint(k, Color.BLACK);
            }
        }


        // Get the series for all intervals and give each a style for rendering
        XYSeriesCollection[] temp = buildOutlinesAndArea(visualized_block);
        XYSeriesCollection pcBlocks = temp[0];
        XYSeriesCollection pcBlocksArea = temp[1];
        for(int i = 0; i < pcBlocks.getSeriesCount(); i++){
            pcBlockRenderer.setSeriesPaint(i, Color.ORANGE);
            pcBlockAreaRenderer.setSeriesPaint(i, new Color(255, 200, 0, 20));
            pcBlockRenderer.setSeriesStroke(i, strokes[0]);

        }

        // Create chart and plot
        JFreeChart pcChart = ChartsAndPlots.createChart(goodGraphLines, false);
        XYPlot plot = ChartsAndPlots.createHBPlot(pcChart, graphColors[tempBlock.classNum]);
        PC_DomainAndRange(plot);



        // Set renderers and datasets for avg and good/bad graph lines
        plot.setRenderer(0, avgRenderer);
        plot.setDataset(0, avgLines);
        plot.setRenderer(1, pcBlockRenderer);
        plot.setDataset(1, pcBlocks);
        plot.setRenderer(2, pcBlockAreaRenderer);
        plot.setDataset(2, pcBlocksArea);
        plot.setRenderer(3, badLineRenderer);
        plot.setDataset(3, badGraphLines);
        plot.setRenderer(4, goodLineRenderer);
        plot.setDataset(4, goodGraphLines);

        // Create ChartPanel
        ChartPanel chartPanel = new ChartPanel(pcChart);
        chartPanel.setMouseWheelEnabled(true);
        return chartPanel;

    }

    /**
     * XYSeriesCollection[0] in return is outlines, XYSeriesCollection[1] is the areas
     *
     * Plan is to build the series using 2-d Arraylist then convert them into the series objects and return them.
     */
    private XYSeriesCollection[] buildOutlinesAndArea(int visualized_block){
        HyperBlock tempBlock = hyper_blocks.get(visualized_block);

        // 2-d arraylist instead of [][] because it is sparse
        ArrayList<ArrayList<Double>> outlineList = new ArrayList<>();
        ArrayList<ArrayList<Double>> areaList = new ArrayList<>();
        int numMaxIntv = tempBlock.getMaxDisjunctiveORs();
        for(int i = 0; i < numMaxIntv; i++){
            // Set up all the rows for the intervals.
            outlineList.add(i, new ArrayList<>(Collections.nCopies(DV.fieldLength * 2, null)));
            areaList.add(i, new ArrayList<>(Collections.nCopies(DV.fieldLength * 2, null)));
        }

        // If the block has more than 1 datapoint in it.
        if (tempBlock.hyper_block.get(0).size() > 1)
        {
            for (int attr = 0; attr < DV.fieldLength; attr++) {
                for (int intv = 0; intv < tempBlock.intervalCount(attr); intv++) {
                    // Add the minimums for each interval to their respective lists
                    outlineList.get(intv).set(attr, tempBlock.minimums.get(attr).get(intv));
                    areaList.get(intv).set(attr, tempBlock.minimums.get(attr).get(intv));
                }
            }

            for (int attr = DV.fieldLength - 1; attr > -1; attr--) {
                for (int intv = 0; intv < tempBlock.intervalCount(attr); intv++) {
                    outlineList.get(intv).set(DV.fieldLength + attr, tempBlock.maximums.get(attr).get(intv));
                    areaList.get(intv).set(DV.fieldLength + attr, tempBlock.maximums.get(attr).get(intv));
                }
            }
        }

        // Length of a row is 2 * Field_length
        XYSeriesCollection outlines = new XYSeriesCollection();
        XYSeriesCollection areas = new XYSeriesCollection();

        // each row is an interval collection, the column number should be the x for first Dv.FieldLengtht
        for(int i = 0; i < numMaxIntv; i++){
            // XYSeries for each interval since it will basically be its own shape.
            XYSeries series = new XYSeries(i, false, true);
            for(int attr = 0; attr < DV.fieldLength; attr++){

                if(outlineList.get(i).get(attr) == null){
                    // Need to add the value from the first interval
                    Double y = outlineList.get(0).get(attr);
                    series.add(attr, y);

                }else{
                    // we can add the value that was in the disjunction
                    series.add(attr, outlineList.get(i).get(attr));
                }
            }

            // Need to go backwards through the elements, but they will be at weird spots since they come after in the list.
            for(int attr = DV.fieldLength - 1; attr > -1; attr--){
                Double value = outlineList.get(i).get(DV.fieldLength + attr);

                if(value == null){
                    value = outlineList.get(0).get(DV.fieldLength + attr);
                }

                series.add(attr, value);
            }

            // Add in the first attributes min to close the shape.
            if(outlineList.get(i).get(0) != null){
                series.add(0, outlineList.get(i).get(0));
            }
            else{
                series.add(0, outlineList.get(0).get(0));
            }

            outlines.addSeries(series);
            areas.addSeries(series);
        }

        /*
        for(int i = 0; i < outlines.getSeriesCount(); i++){
            System.out.println("INTERVAL NUMBER     :     " + i);
            XYSeries series = outlines.getSeries(i);
            XYSeries series2 = areas.getSeries(i);
            System.out.println("THE OUTLINE SERIES IS: ");
            System.out.println(series.getItems());

            System.out.println("THE AREA SERIES IS: ");
            System.out.println(series2.getItems());

        }
         */
        return new XYSeriesCollection[]{outlines, areas};
    }




    private JScrollPane PC_HBs(ArrayList<DataObject> data)
    {
        // create panel with 3 columns
        JPanel tilePanel = new JPanel();
        tilePanel.setLayout(new GridLayout((int)Math.ceil(hyper_blocks.size() / 3.0), 3));
        JScrollPane tileScroll = new JScrollPane(tilePanel);

        // Increases the scroll wheel and up/down arrows to 40 pixels per mouse tick.
        tileScroll.getVerticalScrollBar().setUnitIncrement(40);
        tileScroll.getVerticalScrollBar().setBlockIncrement(40);

        Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize();
        int width = screenSize.width / 3;
        int height = screenSize.height / ((int)Math.ceil(hyper_blocks.size() / 3.0));

        for (int h = 0; h < hyper_blocks.size(); h++)
        {
            // create image from chart
            ChartPanel cPanel = PC_HB(data, h);
            JFreeChart chart = cPanel.getChart();
            BufferedImage image = chart.createBufferedImage(width, height); // width and height cannot be <= 0

            // add image to panel
            JLabel picLabel = new JLabel(new ImageIcon(image));
            tilePanel.add(picLabel);
        }

        return tileScroll;
    }




    public static Color[] getRainbowColors(int n) {
        Color[] colors = new Color[n];

        // Divide the total number of colors into 7 parts
        int segmentLength = n / 7;

        // Define the start and end colors of each rainbow segment
        Color[] rainbowColors = {
                new Color(255, 0, 0),     // Red
                new Color(255, 127, 0),   // Orange
                new Color(255, 255, 0),   // Yellow
                new Color(0, 255, 0),     // Green
                new Color(0, 0, 255),     // Blue
                new Color(75, 0, 130),    // Indigo
                new Color(148, 0, 211)    // Violet
        };

        int index = 0;

        // Generate colors for each segment
        for (int i = 0; i < rainbowColors.length - 1; i++)
        {
            Color startColor = rainbowColors[i];
            Color endColor = rainbowColors[i + 1];

            // Calculate colors between startColor and endColor
            for (int j = 0; j < segmentLength && index < n; j++)
            {
                float ratio = (float) j / segmentLength;

                int red = (int) (startColor.getRed() * (1 - ratio) + endColor.getRed() * ratio);
                int green = (int) (startColor.getGreen() * (1 - ratio) + endColor.getGreen() * ratio);
                int blue = (int) (startColor.getBlue() * (1 - ratio) + endColor.getBlue() * ratio);

                colors[index++] = new Color(red, green, blue);
            }
        }

        // If there's any remaining colors to reach n, fill with the last color (violet)
        while (index < n)
            colors[index++] = rainbowColors[rainbowColors.length - 1];







        return colors;
    }




    private ChartPanel PC_HB_RAINBOW(ArrayList<DataObject> datum, int visualized_block)
    {
        HyperBlock tempBlock = hyper_blocks.get(visualized_block);
        // create main renderer and dataset
        XYLineAndShapeRenderer goodLineRenderer = new XYLineAndShapeRenderer(true, false);
        XYSeriesCollection goodGraphLines = new XYSeriesCollection();
        XYLineAndShapeRenderer badLineRenderer = new XYLineAndShapeRenderer(true, false);
        XYSeriesCollection badGraphLines = new XYSeriesCollection();

        // average renderer and dataset
        XYLineAndShapeRenderer avgRenderer = new XYLineAndShapeRenderer(true, false);
        XYSeriesCollection avgLines = new XYSeriesCollection();

        // hyperblock renderer and dataset
        XYLineAndShapeRenderer pcBlockRenderer = new XYLineAndShapeRenderer(true, false);
        XYSeriesCollection pcBlocks = new XYSeriesCollection();
        XYAreaRenderer pcBlockAreaRenderer = new XYAreaRenderer(XYAreaRenderer.AREA);
        XYSeriesCollection pcBlocksArea = new XYSeriesCollection();

        // create different strokes for each HB within a combined HB
        BasicStroke[] strokes = createStrokes(visualized_block);


        //  get colors
        int clr_cnt = 0;
        for (DataObject data : datum)
        {
            DataVisualization.getCoordinates(datum);
            for (int i = 0; i < data.data.length; i++)
            {
                boolean within = false;
                for (int k = 0; k < tempBlock.hyper_block.size(); k++)
                {
                    if (inside_HB(visualized_block, data.data[i]))
                        within = true;
                }

                if (within)
                {
                    clr_cnt++;
                }
            }
        }

        Color[] lineClrs = getRainbowColors(clr_cnt);
        double[][][] coordinates = new double[clr_cnt][DV.fieldLength][2];
        double[][] linedata = new double[clr_cnt][DV.fieldLength];
        clr_cnt = 0;
        for (DataObject data : datum)
        {
            for (int i = 0; i < data.data.length; i++)
            {
                boolean within = false;
                for (int k = 0; k < tempBlock.hyper_block.size(); k++)
                {
                    if (inside_HB(visualized_block, data.data[i]))
                        within = true;
                }

                if (within)
                {
                    coordinates[clr_cnt] = data.coordinates[i];
                    linedata[clr_cnt++] = data.data[i];
                }
            }
        }

        // sort coordinates
        Arrays.sort(coordinates, (a, b) -> Double.compare(a[DV.fieldLength-1][1], b[DV.fieldLength-1][1]));

        // Get sorted indices based on 3D array sorting
        Integer[] indices = new Integer[coordinates.length];
        for (int i = 0; i < coordinates.length; i++) {
            indices[i] = i;
        }

        // Sort indices based on the 3D array criteria
        Arrays.sort(indices, Comparator.comparingDouble(i -> coordinates[i][DV.fieldLength-1][1]));

        // Create a new sorted 2D array based on sorted indices
        double[][] sortedValues = new double[linedata.length][];
        for (int i = 0; i < indices.length; i++)
            sortedValues[i] = linedata[indices[i]];


        int lineCnt = 0;
        for (int i = 0; i < sortedValues.length; i++)
        {
            // start line at (0, 0)
            XYSeries line = new XYSeries(lineCnt, false, true);

            // add points to lines
            int off = 0;
            for (int j = 0; j < DV.fieldLength; j++)
            {
                if (remove_extra && Math.abs(average_case.get(0).get(0)[j] - average_case.get(1).get(0)[j]) > 0.25)
                {
                    line.add(off, sortedValues[i][j]);
                    off++;
                }
                else if (!remove_extra)
                    line.add(j, sortedValues[i][j]);
            }

            // add series
            goodGraphLines.addSeries(line);
            goodLineRenderer.setSeriesPaint(lineCnt, Color.GREEN);
            goodLineRenderer.setSeriesStroke(lineCnt, strokes[0]);

            lineCnt++;
        }

        // populate average series
        for (int k = 0; k < average_case.get(visualized_block).size(); k++)
        {
            if (tempBlock.hyper_block.get(k).size() > 1)
            {
                XYSeries line = new XYSeries(k, false, true);
                int off = 0;
                for (int i = 0; i < average_case.get(visualized_block).get(k).length; i++)
                    if (remove_extra && Math.abs(average_case.get(0).get(0)[i] - average_case.get(1).get(0)[i]) > 0.25)
                    {
                        line.add(off, average_case.get(visualized_block).get(k)[i]);
                        off++;
                    }
                    else if (!remove_extra)
                        line.add(i, average_case.get(visualized_block).get(k)[i]);

                avgLines.addSeries(line);
                avgRenderer.setSeriesPaint(k, Color.BLACK);
                avgRenderer.setSeriesStroke(k, new BasicStroke(3f, BasicStroke.CAP_BUTT, BasicStroke.JOIN_MITER));


                XYSeries line2 = new XYSeries(5, false, true);
                double[] best = DV.trainData.get(0).data[0];
                double best_dist = Double.MAX_VALUE;

                for (int d = 0; d < DV.trainData.size(); d++)
                {
                    for (int b = 0; b < DV.trainData.get(d).data.length; b++)
                    {
                        double new_dist = Distance.euclidean_distance(average_case.get(k).get(0), DV.trainData.get(d).data[b]);
                        if (new_dist < best_dist)
                        {
                            best = DV.trainData.get(d).data[b];
                            best_dist = new_dist;
                        }
                    }
                }

                for (int w = 0; w < DV.fieldLength; w++)
                {
                    off = 0;
                    for (int i = 0; i < DV.fieldLength; i++)
                        if (remove_extra && Math.abs(average_case.get(0).get(0)[i] - average_case.get(1).get(0)[i]) > 0.25)
                        {
                            line2.add(off, best[i]);
                            off++;
                        }
                        else if (!remove_extra)
                            line2.add(i, best[i]);
                }

                avgLines.addSeries(line2);
                for (int w = 0; w < 10; w++)
                {
                    if (w != k)
                    {
                        avgRenderer.setSeriesPaint(w, Color.RED);
                        avgRenderer.setSeriesStroke(w, new BasicStroke(3f, BasicStroke.CAP_BUTT, BasicStroke.JOIN_MITER));
                    }
                }
            }
        }





        // add hyperblocks
        //AUSTIN: I think this k loop is useless, since each blocks' "hyper_block" is always length 1 rn. 12/19/2024
        for (int k = 0, offset = 0; k < tempBlock.hyper_block.size(); k++)
        {
            if (tempBlock.hyper_block.get(k).size() > 1)
            {
                XYSeries tmp1 = new XYSeries(k-offset, false, true);
                XYSeries tmp2 = new XYSeries(k-offset, false, true);
                int cnt = 0;
                for (int j = 0; j < DV.fieldLength; j++)
                {
                    if (remove_extra && Math.abs(average_case.get(0).get(0)[j] - average_case.get(1).get(0)[j]) > 0.25)
                    {
                        tmp1.add(cnt, tempBlock.minimums.get(j).get(0));
                        tmp2.add(cnt, tempBlock.minimums.get(j).get(0));
                        cnt++;
                    }
                    else if (!remove_extra)
                    {
                        tmp1.add(j, tempBlock.minimums.get(j).get(0));
                        tmp2.add(j, tempBlock.minimums.get(j).get(0));
                    }
                }

                for (int j = DV.fieldLength - 1; j > -1; j--)
                {
                    if (remove_extra && Math.abs(average_case.get(0).get(0)[j] - average_case.get(1).get(0)[j]) > 0.25)
                    {
                        tmp1.add(cnt, tempBlock.minimums.get(j).get(0));
                        tmp2.add(cnt, tempBlock.minimums.get(j).get(0));
                        cnt--;
                    }
                    else if (!remove_extra)
                    {
                        tmp1.add(j, tempBlock.minimums.get(j).get(0));
                        tmp2.add(j, tempBlock.minimums.get(j).get(0));
                    }
                }

                tmp1.add(0, tempBlock.minimums.get(0).get(0));
                tmp2.add(0, tempBlock.minimums.get(0).get(0));

                pcBlockRenderer.setSeriesPaint(k-offset, Color.ORANGE);
                pcBlockAreaRenderer.setSeriesPaint(k-offset, new Color(255, 200, 0, 20));
                pcBlockRenderer.setSeriesStroke(k, strokes[k]);

                pcBlocks.addSeries(tmp1);
                pcBlocksArea.addSeries(tmp2);
            }
            else
                offset++;
        }

        // create chart and plot
        JFreeChart pcChart = ChartsAndPlots.createChart(goodGraphLines, false);
        XYPlot plot = ChartsAndPlots.createHBPlot(pcChart, graphColors[tempBlock.classNum]);
        PC_DomainAndRange(plot);

        // set renderers and datasets
        plot.setRenderer(0, avgRenderer);
        plot.setDataset(0, avgLines);
        plot.setRenderer(1, pcBlockRenderer);
        plot.setDataset(1, pcBlocks);
        plot.setRenderer(2, pcBlockAreaRenderer);
        plot.setDataset(2, pcBlocksArea);
        plot.setRenderer(3, badLineRenderer);
        plot.setDataset(3, badGraphLines);
        plot.setRenderer(4, goodLineRenderer);
        plot.setDataset(4, goodGraphLines);

        ChartPanel chartPanel = new ChartPanel(pcChart);
        chartPanel.setMouseWheelEnabled(true);
        return chartPanel;
    }





    private ChartPanel GLC_L_HB(ArrayList<DataObject> datum, int visualized_block)
    {
        HyperBlock tempBlock = hyper_blocks.get(visualized_block);
        // create main renderer and dataset
        XYLineAndShapeRenderer goodLineRenderer = new XYLineAndShapeRenderer(true, false);
        XYSeriesCollection goodGraphLines = new XYSeriesCollection();
        XYLineAndShapeRenderer badLineRenderer = new XYLineAndShapeRenderer(true, false);
        XYSeriesCollection badGraphLines = new XYSeriesCollection();

        XYLineAndShapeRenderer endpointRenderer = new XYLineAndShapeRenderer(false, true);
        XYSeriesCollection endpoints = new XYSeriesCollection();;

        // average renderer and dataset
        XYLineAndShapeRenderer avgRenderer = new XYLineAndShapeRenderer(true, false);
        XYSeriesCollection avgLines = new XYSeriesCollection();

        // hyperblock renderer and dataset
        XYLineAndShapeRenderer pcBlockRenderer = new XYLineAndShapeRenderer(true, false);
        XYSeriesCollection pcBlocks = new XYSeriesCollection();
        XYAreaRenderer pcBlockAreaRenderer = new XYAreaRenderer(XYAreaRenderer.AREA);
        XYSeriesCollection pcBlocksArea = new XYSeriesCollection();

        // create different strokes for each HB within a combined HB
        BasicStroke[] strokes = createStrokes(visualized_block);
        Ellipse2D.Double endpointShape = new Ellipse2D.Double(-1, -1, 2, 2);


        //  get colors
        int clr_cnt = 0;
        for (DataObject data : datum)
        {
            DataVisualization.getCoordinates(datum);
            for (int i = 0; i < data.data.length; i++)
            {
                // If the current datapoint is within the block.
                if (inside_HB(visualized_block,  data.data[i]))
                {
                    clr_cnt++;
                }
            }
        }

        //Color[] lineClrs = getRainbowColors(clr_cnt);
        double[][][] coordinates = new double[clr_cnt][DV.fieldLength][2];
        clr_cnt = 0;
        for (DataObject data : datum)
        {
            for (int i = 0; i < data.data.length; i++)
            {
                boolean within = false;
                for (int k = 0; k < tempBlock.hyper_block.size(); k++)
                {
                    if (inside_HB(visualized_block,  data.data[i]))
                        within = true;
                }

                if (within)
                    coordinates[clr_cnt++] = data.coordinates[i];
            }
        }

        // sort coordinates
        //Arrays.sort(coordinates, (a, b) -> Double.compare(a[DV.fieldLength-1][1], b[DV.fieldLength-1][1]));


        int lineCnt = -1;
        double buffer = DV.fieldLength / 10.0;
        for (double[][] data : coordinates)
        {
            int upOrDown = 1;

            // start line at (0, 0)
            XYSeries line;
            XYSeries endpointSeries;

            line = new XYSeries(++lineCnt, false, true);
            endpointSeries = new XYSeries(lineCnt, false, true);

            line.setNotify(false);
            endpointSeries.setNotify(false);

            if (DV.showFirstSeg)
                line.add(0, upOrDown * buffer);

            // add points to lines
            for (int j = 0; j < data.length; j++)
            {
                line.add(data[j][0], upOrDown * (data[j][1] + buffer));

                // add endpoint and timeline
                if (j == data.length - 1)
                {
                    endpointSeries.add(data[j][0], upOrDown * (data[j][1] + buffer));

                    goodGraphLines.addSeries(line);
                    goodLineRenderer.setSeriesPaint(lineCnt, graphColors[tempBlock.classNum]);//lineClrs[lineCnt]);
                    goodLineRenderer.setSeriesStroke(lineCnt, strokes[0]);

                    endpoints.addSeries(endpointSeries);
                    endpointRenderer.setSeriesShape(lineCnt, endpointShape);
                    endpointRenderer.setSeriesPaint(lineCnt, DV.endpoints);
                }
            }
        }

        // populate average series
        for (int k = 0; k < average_case.get(visualized_block).size(); k++)
        {
            if (tempBlock.hyper_block.get(k).size() > 1)
            {
                XYSeries line = new XYSeries(k, false, true);
                for (int i = 0; i < average_case.get(visualized_block).get(k).length; i++)
                    if (remove_extra && !Objects.equals(tempBlock.minimums.get(i).get(0), tempBlock.maximums.get(i).get(0)))
                        line.add(i, average_case.get(visualized_block).get(k)[i]);

                avgLines.addSeries(line);
                avgRenderer.setSeriesPaint(k, Color.BLACK);
            }
        }

        // add hyperblocks
        for (int k = 0, offset = 0; k < tempBlock.hyper_block.size(); k++)
        {
            if (tempBlock.hyper_block.get(k).size() > 1)
            {
                XYSeries tmp1 = new XYSeries(k-offset, false, true);
                XYSeries tmp2 = new XYSeries(k-offset, false, true);

                for (int j = 0; j < DV.fieldLength; j++)
                {
                    if (remove_extra && !Objects.equals(tempBlock.minimums.get(j).get(0), tempBlock.maximums.get(j).get(0)))
                    {

                        tmp1.add(j, tempBlock.minimums.get(j).get(0));
                        tmp2.add(j, tempBlock.minimums.get(j).get(0));
                    }
                }

                for (int j = DV.fieldLength - 1; j > -1; j--)
                {
                    if (remove_extra && !Objects.equals(tempBlock.minimums.get(j).get(0), tempBlock.maximums.get(j).get(0)))
                    {
                        tmp1.add(j, tempBlock.maximums.get(j).get(0));
                        tmp2.add(j, tempBlock.maximums.get(j).get(0));
                    }
                }

                for (int j = 0; j < DV.fieldLength; j++)
                {
                    if (remove_extra && !Objects.equals(tempBlock.minimums.get(j).get(0), tempBlock.maximums.get(j).get(0)))
                    {
                        tmp1.add(j, tempBlock.maximums.get(j).get(0));
                        tmp2.add(j, tempBlock.maximums.get(j).get(0));
                        break;
                    }
                }
                //tmp1.add(0, tempBlock.minimums.get(k)[0]);
                //tmp2.add(0, tempBlock.minimums.get(k)[0]);

                pcBlockRenderer.setSeriesPaint(k-offset, Color.ORANGE);
                pcBlockAreaRenderer.setSeriesPaint(k-offset, new Color(255, 200, 0, 20));
                pcBlockRenderer.setSeriesStroke(k, strokes[k]);

                pcBlocks.addSeries(tmp1);
                pcBlocksArea.addSeries(tmp2);
            }
            else
                offset++;
        }

        // create chart and plot
        JFreeChart pcChart = ChartsAndPlots.createChart(goodGraphLines, false);
        XYPlot plot = ChartsAndPlots.createPlot(pcChart, tempBlock.classNum);
        PC_DomainAndRange(plot);

        // set renderers and datasets
        plot.setRenderer(0, avgRenderer);
        plot.setDataset(0, avgLines);
        plot.setRenderer(1, pcBlockRenderer);
        plot.setDataset(1, pcBlocks);
        plot.setRenderer(2, endpointRenderer);
        plot.setDataset(2, endpoints);
        plot.setRenderer(4, goodLineRenderer);
        plot.setDataset(4, goodGraphLines);

        ChartPanel chartPanel = new ChartPanel(pcChart);
        chartPanel.setMouseWheelEnabled(true);
        chartPanel.restoreAutoBounds();
        return chartPanel;
    }


    /**
     *                          *
     * ------------------------ *
     * UNSTABLE FUNCTIONS BELOW *
     * ------------------------ *
     *                          *
     */



    //
    ////
    ////// DATA GENERALIZATION TESTING
    ////
    //


    private void findBestAttribute()
    {
        for (int i = 0; i < DV.angles.length; i++)
        {
            if (DV.angles[i] > DV.angles[best_attribute])
                best_attribute = i;
        }
    }


    private static void sortByColumn(ArrayList<double[]> arr, int col)
    {
        // Using built-in sort function Arrays.sort with lambda expressions
        arr.sort(Comparator.comparingDouble(o -> o[col])); // increasing order
    }

    private ArrayList<double[]> find_envelope_cases(int num)
    {
        HyperBlock tempBlock = hyper_blocks.get(num);
        ArrayList<double[]> cases = new ArrayList<>();

        boolean[] max = new boolean[DV.fieldLength];
        boolean[] min = new boolean[DV.fieldLength];

        for (int i = 0; i < DV.fieldLength; i++)
        {
            max[i] = true;
            min[i] = true;
        }

        for (int i = 0; i < tempBlock.hyper_block.get(0).size(); i++)
        {
            boolean part = false;

            for (int j = 0; j < DV.fieldLength; j++)
            {
                if (tempBlock.hyper_block.get(0).get(i)[j] == tempBlock.maximums.get(j).get(0) && max[j])
                {
                    part = true;
                    max[j] = false;
                }
                else if (tempBlock.hyper_block.get(0).get(i)[j] == tempBlock.minimums.get(j).get(0) && min[j])
                {
                    part = true;
                    min[j] = false;
                }
            }

            if (part)
                cases.add(tempBlock.hyper_block.get(0).get(i));
        }

        return cases;
    }


    private boolean kNN(double[] givenCase, int k, int mis_class)
    {
        // Create a list to store distances and corresponding hyperblocks
        List<Map.Entry<Double, HyperBlock>> distances = new ArrayList<>();

        // Calculate the distance between the given case and each hyperblock
        for (int i = 0; i < hyper_blocks.size(); i++)
        {
            double[] lossless_distance = Distance.lossless_distance(givenCase, average_case.get(i).get(0));
            double distance = 0;
            for (double v : lossless_distance) distance += v;

            distances.add(new AbstractMap.SimpleEntry<>(distance, hyper_blocks.get(i)));
        }

        // Sort the hyperblocks based on the calculated distances
        distances.sort(Map.Entry.comparingByKey());

        // Select the k-nearest hyperblocks
        List<HyperBlock> nearestHyperBlocks = new ArrayList<>();
        for (int i = 0; i < k && i < distances.size(); i++)
            nearestHyperBlocks.add(distances.get(i).getValue());

        // Output the nearest hyperblocks (for demonstration purposes)
        /*for (HyperBlock hb : nearestHyperBlocks)
            System.out.println("\tNearest HyperBlock: " + hb.classNum);*/

        // Determine the majority class of the k-nearest hyperblocks
        int[] class_cnts = new int[DV.classNumber];

        for (HyperBlock hb : nearestHyperBlocks)
            class_cnts[hb.classNum]++;

        int maj = 0;
        for (int i = 0; i < DV.classNumber; i++)
            if (class_cnts[i] > class_cnts[maj])
                maj = i;

        if (mis_class == maj)
        {
            //System.out.println("Result: correct classification");
            return true;
        }
        else
        {
            //System.out.println("Result: misclassification");
            return false;
        }
    }


    /**
     * Orders hyperblocks by size and class
     */
    private void order_hbs_by_class()
    {
        ArrayList<ArrayList<HyperBlock>> ordered = new ArrayList<>();
        for (int i = 0; i < DV.classNumber; i++)
            ordered.add(new ArrayList<>());

        for (HyperBlock hyperBlock : hyper_blocks)
            ordered.get(hyperBlock.classNum).add(hyperBlock);

        hyper_blocks.clear();
        for (ArrayList<HyperBlock> hyperBlocks : ordered)
        {
            hyperBlocks.sort(new HBComparator());
            hyper_blocks.addAll(hyperBlocks);
        }
    }

    private void sort_hb_by_size(){
        hyper_blocks.sort(new HBComparator());
    }


    private void HB_analytics()
    {
        average_case.clear();

        // count cases in HB and not in any HBs
        int[] counter = new int[hyper_blocks.size()];
        int not_in = 0;
        double global_maj = 0;
        int global_cnt = 0;

        for (int h = 0; h < hyper_blocks.size(); h++)
        {
            for (int q = 0; q < hyper_blocks.get(h).hyper_block.size(); q++)
                hyper_blocks.get(h).hyper_block.get(q).clear();

            double maj_cnt = 0;
            for (int i = 0; i < data.size(); i++)
            {
                for (int j = 0; j < data.get(i).data.length; j++)
                {
                    for (int q = 0; q < hyper_blocks.get(h).hyper_block.size(); q++)
                    {
                        if (inside_HB(h, data.get(i).data[j]))
                        {
                            if (i == hyper_blocks.get(h).classNum)
                            {
                                maj_cnt++;
                                global_maj++;
                            }


                            counter[h]++;
                            global_cnt++;
                            hyper_blocks.get(h).hyper_block.get(q).add(data.get(i).data[j]);
                        }
                    }
                }
            }

            //System.out.println("\nBlock " + (h+1) + " Size: " + counter[h]);
            //System.out.println("Block " + (h+1) + " Accuracy: " + (maj_cnt / counter[h]));

            accuracy.add(maj_cnt / counter[h]);
            misclassified.add(counter[h] - (int) maj_cnt);
            hyper_blocks.get(h).size = counter[h];

            ArrayList<double[]> avg_cases = new ArrayList<>();
            for (int q = 0; q < hyper_blocks.get(h).hyper_block.size(); q++)
            {
                double[] avg = new double[DV.fieldLength];
                Arrays.fill(avg, 0);

                for (int j = 0; j < hyper_blocks.get(h).hyper_block.get(q).size(); j++)
                {
                    for (int k = 0; k < hyper_blocks.get(h).hyper_block.get(q).get(j).length; k++)
                        avg[k] += hyper_blocks.get(h).hyper_block.get(q).get(j)[k];
                }

                for (int j = 0; j < hyper_blocks.get(h).hyper_block.get(q).get(0).length; j++)
                    avg[j] /= hyper_blocks.get(h).hyper_block.get(q).size();

                avg_cases.add(avg);
            }

            average_case.add(avg_cases);
        }

        int bcnt = 0;
        for (HyperBlock hyperBlock : hyper_blocks)
        {
            for (int q = 0; q < hyperBlock.hyper_block.size(); q++)
                bcnt += hyperBlock.hyper_block.get(q).size();
        }

        //System.out.println("TOTAL NUM IN BLOCKS: " + bcnt);

        for (int i = 0; i < data.size(); i++)
        {
            for (int j = 0; j < data.get(i).data.length; j++)
            {
                boolean in = false;
                for (int h = 0; h < hyper_blocks.size(); h++)
                {
                    for (int q = 0; q < hyper_blocks.get(h).hyper_block.size(); q++)
                    {
                        if (inside_HB(h, data.get(i).data[j]))
                        {
                            in = true;
                            break;
                        }
                    }
                }

                if (!in)
                    not_in++;
            }
        }

        //System.out.println("NOT IN ANY BLOCKS: " + not_in + "\n");
        //System.out.println("Total Accuracy: " + (global_maj / global_cnt));
    }

    //TODO: MAKE SURE CLASS CHECK NOT NEEDED
    public void removeUselessBlocks(){

        // Keep track of distinct points in each block.
        int[] in = new int[hyper_blocks.size()];

        for(int i = 0; i < data.size(); i++){
            // Go through each data point
            for(int j = 0; j < data.get(i).data.length; j++){
                double[] point = data.get(i).data[j];

                ArrayList<Integer> potentialHomes = new ArrayList<>();

                // Go through all blocks and let them claim a point
                for(int hb = 0; hb < hyper_blocks.size(); hb++){
                    // If it is inside a block, let them claim it and keep away from other blocks.

                    if(inside_HB(hb, point)){

                        potentialHomes.add(hb);
                    }
                }

                int bestBlock = -1;
                int biggestBlock = -1;
                for (Integer blockToChoose : potentialHomes){
                    if (in[blockToChoose] > biggestBlock) {
                        bestBlock = blockToChoose;
                        biggestBlock = in[blockToChoose];
                    }
                }

                if(bestBlock >= 0){
                    in[bestBlock]++;
                }
            }
        }


        while(true) {
            // this point wants to go into blocks 2, 3, 4. pick whichever has biggest in value now.
            // this way, if a large block missed out on a point previously, it gets a second chance
            int size = hyper_blocks.size();
            int[] in_new = new int[size];

            // if a block gets deleted, run it again.
            // run until no blocks can be deleted.

            /////
            for(int i = 0; i < data.size(); i++){
                // Go through each data point
                for(int j = 0; j < data.get(i).data.length; j++){
                    double[] point = data.get(i).data[j];

                    ArrayList<Integer> potentialHomes = new ArrayList<>();

                    // Go through all blocks and let them claim a point
                    for(int hb = 0; hb < hyper_blocks.size(); hb++){
                        // If it is inside a block, let them claim it and keep away from other blocks.

                        if(inside_HB(hb, point)){
                            potentialHomes.add(hb);
                        }
                    }


                    int bestBlock = -1;
                    int biggestBlock = -1;
                    for (Integer blockToChoose : potentialHomes){
                        if (in[blockToChoose] > biggestBlock) {
                            bestBlock = blockToChoose;
                            biggestBlock = in[blockToChoose];
                        }
                    }

                    if(bestBlock >= 0){
                        in_new[bestBlock]++;
                    }
                }
            }
            /////

            for (int i = in_new.length - 1; i > -1; i--) {
                if (in_new[i] == 0) {
                    hyper_blocks.remove(i);
                }
            }

            if (hyper_blocks.size() == size) {
                break;
            }

            in = in_new;
        }
    }

    /**
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

    private boolean inside_HB_Intervals(ArrayList<ArrayList<Double>> minimums, ArrayList<ArrayList<Double>> maximums, double[] point, int exclude){
        boolean inside = true;

        // Go through all attributes
        for (int i = 0; i < DV.fieldLength; i++)
        {
            if(exclude == i){
                continue;
            }

            boolean inAnInterval = false;

            // Go through all intervals the hyperblock allows for the attribute
            for(int j = 0; j < maximums.get(i).size(); j++){
                // If the datapoints value falls inside one of the intervals.
                if (point[i] >= minimums.get(i).get(j) && point[i] <= maximums.get(i).get(j)) {
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
     * Creates strokes for hyperblocks
     * @param hb Hyperblock number
     * @return Strokes for hyperblocks
     */
    private BasicStroke[] createStrokes(int hb)
    {
        BasicStroke[] strokes = new BasicStroke[hyper_blocks.get(hb).hyper_block.size()];
        for (int k = 0; k < hyper_blocks.get(hb).hyper_block.size(); k++)
        {
            if (k == 0)
                strokes[k] = new BasicStroke(3f, BasicStroke.CAP_BUTT, BasicStroke.JOIN_MITER);
            else
            {
                float dashLength = 5f * k; // Increase dash length progressively
                float[] dashPattern = {dashLength, dashLength}; // Dash and space are the same length
                strokes[k] = new BasicStroke(3f, BasicStroke.CAP_BUTT, BasicStroke.JOIN_MITER, 1f, dashPattern, 0f);
            }
        }

        return strokes;
    }


    /**
     * Sets the domain and range for a parallel coordinate plot
     * @param plot Plot to set domain and range
     */
    private void PC_DomainAndRange(XYPlot plot)
    {
        // set domain
        ValueAxis domainView = plot.getDomainAxis();
        domainView.setRange(-0.1, DV.fieldLength-1);

        // create domain and range ticks
        int tick = DV.fieldLength > 10 ? DV.fieldLength / 10 : 1;
        NumberAxis xAxis = (NumberAxis) plot.getDomainAxis();
        NumberTickUnit ntu = new NumberTickUnit(tick)
        {
            @Override
            public String valueToString(double value) {
                return super.valueToString(value + 1);
            }
        };
        xAxis.setTickUnit(ntu);

        // set range
        NumberAxis yAxis = (NumberAxis) plot.getRangeAxis();
        yAxis.setTickUnit(new NumberTickUnit(0.25));
        yAxis.setAutoRange(false);
        yAxis.setRange(0, 1);
    }


    /**
     * Tests hyperblocks with test data
     */
    public void test_HBs()
    {
        ArrayList<Double> testCaseAcc = new ArrayList<>();
        double globalGood = 0;
        double globalBad = 0;
        double classifiedCases = 0;
        double totalCases = 0;


        ArrayList<double[]> missed_cases = new ArrayList<>();
        ArrayList<Integer> missed_cases_classes = new ArrayList<>();

        ArrayList<double[]> bad_cases = new ArrayList<>();
        ArrayList<Integer> bad_cases_classes = new ArrayList<>();


        ArrayList<Integer> testCaseMis = new ArrayList<>();
        for (int i = 0; i < DV.testData.size(); i++)
            testCaseMis.add(0);

        for (int i = 0; i < DV.testData.size(); i++)
        {
            double good = 0;
            double bad = 0;

            for (int j = 0; j < DV.testData.get(i).data.length; j++)
            {
                boolean classified = false;
                totalCases++;

                for (int hbIndex = 0; hbIndex < hyper_blocks.size(); hbIndex++)
                {
                    // Check if the current point is within the current hyper-block
                    boolean inside = inside_HB(hbIndex, DV.testData.get(i).data[j]);

                    if (inside)
                    {
                        classified = true;

                        if (i == hyper_blocks.get(hbIndex).classNum)
                            good++;
                        else
                        {
                            bad++;
                            bad_cases.add(DV.testData.get(i).data[j]);
                            bad_cases_classes.add(i);
                        }
                    }
                }

                if (classified)
                    classifiedCases++;
                else
                {
                    testCaseMis.set(i, testCaseMis.get(i) + 1);
                    missed_cases.add(DV.testData.get(i).data[j]);
                    missed_cases_classes.add(i);
                }

            }

            testCaseAcc.add(good / (good + bad));
            globalGood += good;
            globalBad += bad;
        }

        System.out.println("\n\nRESULTS OF TEST CASES: ");
        System.out.printf("\tGlobal Accuracy: %.4f%%%n", (globalGood / (globalGood + globalBad)) * 100);
        System.out.println("\tTotal Cases Classified: " + (int)classifiedCases + " / " + (int)totalCases);
        System.out.printf("\tPercentage of Cases Classified: %.4f%%%n", (classifiedCases / totalCases) * 100);

        System.out.println("\nCases not classified by class:");
        for (int i = 0; i < testCaseMis.size(); i++)
            System.out.println("\tClass " + (i) + " Not Classified: " + testCaseMis.get(i));

        System.out.println("\nTest Case Accuracy by class:");
        for (int i = 0; i < testCaseAcc.size(); i++)
            System.out.printf("\tClass " + (i) + " Test Case Accuracy: %.4f%%%n", testCaseAcc.get(i) * 100);


        /*for (int i = 0; i < bad_cases.size(); i++)
        {
            System.out.println("\nMisclassified Case Class:" + bad_cases_classes.get(i));
            kNN(bad_cases.get(i), 5);
        }*/

        // print each missed case
        for (int i = 0; i < missed_cases.size(); i++)
        {
            //System.out.println("\nNot Classified Case Class:" + missed_cases_classes.get(i));
            boolean res = kNN(missed_cases.get(i), 5, missed_cases_classes.get(i));
            if (res)
                globalGood++;
            else
                globalBad++;
            classifiedCases++;
        }

        System.out.println("\n\nRESULTS OF TEST CASES AFTER VOTING: ");
        System.out.printf("\tGlobal Accuracy: %.4f%%%n", (globalGood / (globalGood + globalBad)) * 100);
        System.out.println("\tTotal Cases Classified: " + (int)classifiedCases + " / " + (int)totalCases);
        System.out.printf("\tPercentage of Cases Classified: %.4f%%%n", (classifiedCases / totalCases) * 100);
    }
















    //TODO:AUSTIN: Get this working with disjunctive blocks. right now it is just reformatted to work with new minimums and maximums format
    private void merger_save(ExecutorService executorService, ArrayList<ArrayList<double[]>> data, ArrayList<ArrayList<double[]>> out_data)
    {
        ArrayList<HyperBlock> blocks = new ArrayList<>(hyper_blocks);
        hyper_blocks.clear();

        for (int i = 0, cnt = blocks.size(); i < out_data.size(); i++)
        {
            // create hyperblock from each datapoint
            for (int j = 0; j < out_data.get(i).size(); j++)
            {
                blocks.add(new HyperBlock(new ArrayList<>(), i));
                blocks.get(cnt).hyper_block.add(new ArrayList<>(List.of(out_data.get(i).get(j))));
                blocks.get(cnt).findBounds();
                cnt++;
            }
        }

        boolean actionTaken;
        ArrayList<Integer> toBeDeleted = new ArrayList<>();
        int cnt = blocks.size();

        System.out.println("Size before merge: " + blocks.size());

        do
        {
            if (cnt <= 0)
            {
                cnt = blocks.size();
            }

            toBeDeleted.clear();
            actionTaken = false;

            if (blocks.size() <= 0)
            {
                break;
            }

            HyperBlock tmp = blocks.get(0);
            blocks.remove(0);

            int tmpClass = tmp.classNum;

            for (int i = 0; i < blocks.size(); i++)
            {
                int curClass = blocks.get(i).classNum;

                if (tmpClass != curClass)
                    continue;

                ArrayList<Double> maxPoint = new ArrayList<>();
                ArrayList<Double> minPoint = new ArrayList<>();

                // define combined space
                for (int j = 0; j < DV.fieldLength; j++)
                {
                    double newLocalMax = Math.max(tmp.maximums.get(j).get(0), blocks.get(i).maximums.get(j).get(0));
                    double newLocalMin = Math.min(tmp.minimums.get(j).get(0), blocks.get(i).minimums.get(j).get(0));

                    maxPoint.add(newLocalMax);
                    minPoint.add(newLocalMin);
                }

                ArrayList<double[]> pointsInSpace = new ArrayList<>();
                ArrayList<Integer> classInSpace = new ArrayList<>();

                for (int j = 0; j < data.size(); j++)
                {
                    for (int k = 0; k < data.get(j).size(); k++)
                    {
                        boolean withinSpace = true;
                        double[] tmp_pnt = new double[DV.fieldLength];

                        for (int w = 0; w < DV.fieldLength; w++)
                        {
                            tmp_pnt[w] = data.get(j).get(k)[w];

                            if (!(tmp_pnt[w] <= maxPoint.get(w) && tmp_pnt[w] >= minPoint.get(w)))
                            {
                                withinSpace = false;
                                break;
                            }
                        }

                        if (withinSpace)
                        {
                            pointsInSpace.add(tmp_pnt);
                            classInSpace.add(j);
                        }
                    }
                }

                // check if new space is pure
                HashSet<Integer> classCnt = new HashSet<>(classInSpace);

                if (classCnt.size() <= 1)
                {
                    actionTaken = true;
                    tmp.hyper_block.get(0).clear();
                    tmp.hyper_block.get(0).addAll(pointsInSpace);
                    tmp.findBounds();
                    toBeDeleted.add(i);
                }
            }

            int offset = 0;

            for (int i : toBeDeleted)
            {
                blocks.remove(i-offset);
                offset++;
            }

            blocks.add(tmp);
            cnt--;

        } while (actionTaken || cnt > 0);

        System.out.println("Size after first merge: " + blocks.size());

        // impure
        cnt = blocks.size();

        do
        {
            if (cnt <= 0)
            {
                cnt = blocks.size();
            }

            toBeDeleted.clear();
            actionTaken = false;

            if (blocks.size() <= 1)
            {
                break;
            }

            HyperBlock tmp = blocks.get(0);
            blocks.remove(0);

            ArrayList<Double> acc = new ArrayList<>();

            for (HyperBlock block : blocks)
            {
                double[] maxPoint = new double[DV.fieldLength];
                double[] minPoint = new double[DV.fieldLength];

                // define combined space
                for (int j = 0; j < DV.fieldLength; j++)
                {
                    double newLocalMax = Math.max(tmp.maximums.get(j).get(0), block.maximums.get(j).get(0));
                    double newLocalMin = Math.min(tmp.minimums.get(j).get(0), block.minimums.get(j).get(0));

                    maxPoint[j] = newLocalMax;
                    minPoint[j] = newLocalMin;
                }

                ArrayList<Integer> classInSpace = new ArrayList<>();
                ArrayList<ArrayList<double[]>> pointsInSpace = new ArrayList<>();

                for (int j = 0; j < data.size(); j++)
                {
                    pointsInSpace.add(new ArrayList<>());

                    for (int k = 0; k < data.get(j).size(); k++)
                    {
                        boolean withinSpace = true;
                        double[] tmp_pnt = new double[DV.fieldLength];

                        for (int w = 0; w < DV.fieldLength; w++)
                        {
                            tmp_pnt[w] = data.get(j).get(k)[w];

                            if (!(tmp_pnt[w] <= maxPoint[w] && tmp_pnt[w] >= minPoint[w]))
                            {
                                withinSpace = false;
                                break;
                            }
                        }

                        if (withinSpace)
                        {
                            classInSpace.add(j);
                            pointsInSpace.get(j).add(tmp_pnt);
                        }

                    }
                }

                HashMap<Integer, Integer> classCnt = new HashMap<>();

                // check if new space is pure enough
                for (int ints : classInSpace)
                {
                    if (classCnt.containsKey(ints))
                        classCnt.replace(ints, classCnt.get(ints) + 1);
                    else
                        classCnt.put(ints, 1);
                }

                int majorityClass = 0;

                for (int key : classCnt.keySet())
                {
                    if (classCnt.get(key) > classCnt.get(majorityClass))
                        majorityClass = key;
                }

                double curClassTotal = 0;
                double classTotal = 0;

                for (int key : classCnt.keySet())
                {
                    if (key == majorityClass)
                        curClassTotal = classCnt.get(key);

                    classTotal += classCnt.get(key);
                }

                double cur_acc = curClassTotal / classTotal;

                // check if a point misclassified by the LDF lies in the new space
                if (explain_ldf)
                {
                    // collect misclassified data within range
                    ArrayList<ArrayList<double[]>> mis_in_range = misclassified_by_ldf_in_range(minPoint, maxPoint);

                    // ensure that data classified correctly by LDF is also correct in HB
                    // only check data not in majority (not classified correctly)
                    for (int i = 0; i < pointsInSpace.size(); i++)
                    {
                        if (i != majorityClass)
                        {
                            for (int j = 0; j < pointsInSpace.get(i).size(); j++)
                            {
                                boolean misclassified_by_both = false;

                                for (int k = 0; k < mis_in_range.get(i).size(); k++)
                                {
                                    if (Arrays.equals(pointsInSpace.get(i).get(j), mis_in_range.get(i).get(k)))
                                    {
                                        misclassified_by_both = true;
                                        break;
                                    }
                                }

                                if (!misclassified_by_both)
                                {
                                    cur_acc = -1;
                                    break;
                                }
                            }
                        }

                        if (cur_acc == -1)
                            break;
                    }
                }

                acc.add(cur_acc);
            }

            int highestAccIndex = 0;

            for (int j = 0; j < acc.size(); j++)
            {
                if (acc.get(j) > acc.get(highestAccIndex))
                    highestAccIndex = j;
            }

            // if acc meets threshold
            if (acc.get(highestAccIndex) >= acc_threshold)
            {
                actionTaken = true;

                ArrayList<Double> maxPoint = new ArrayList<>();
                ArrayList<Double> minPoint = new ArrayList<>();

                // define combined space
                for (int j = 0; j < DV.fieldLength; j++)
                {
                    double newLocalMax = Math.max(tmp.maximums.get(j).get(0), blocks.get(highestAccIndex).maximums.get(j).get(0));
                    double newLocalMin = Math.min(tmp.minimums.get(j).get(0), blocks.get(highestAccIndex).minimums.get(j).get(0));

                    maxPoint.add(newLocalMax);
                    minPoint.add(newLocalMin);
                }

                ArrayList<double[]> pointsInSpace = new ArrayList<>();

                for (ArrayList<double[]> datum : data)
                {
                    for (double[] doubles : datum)
                    {
                        boolean withinSpace = true;
                        double[] tmp_pnt = new double[DV.fieldLength];

                        for (int w = 0; w < DV.fieldLength; w++)
                        {
                            tmp_pnt[w] = doubles[w];

                            if (!(tmp_pnt[w] <= maxPoint.get(w) && tmp_pnt[w] >= minPoint.get(w)))
                            {
                                withinSpace = false;
                                break;
                            }
                        }

                        if (withinSpace)
                            pointsInSpace.add(tmp_pnt);
                    }
                }

                if (tmp.hyper_block.get(0).size() < blocks.get(highestAccIndex).hyper_block.get(0).size())
                    tmp.classNum = blocks.get(highestAccIndex).classNum;

                tmp.hyper_block.get(0).clear();
                tmp.hyper_block.get(0).addAll(pointsInSpace);
                tmp.findBounds();

                // store this index to delete the cube that was combined
                toBeDeleted.add(highestAccIndex);
            }

            int offset = 0;

            for (int i : toBeDeleted)
            {
                blocks.remove(i-offset);
                offset++;
            }

            blocks.add(tmp);
            cnt--;

        } while (actionTaken || cnt > 0);

        System.out.println("Size after second merge: " + blocks.size());

        hyper_blocks.addAll(blocks);
    }



    private void merger_cuda(ArrayList<ArrayList<double[]>> data, ArrayList<ArrayList<double[]>> out_data) throws ExecutionException, InterruptedException
    {
        // Initialize JCuda
        JCudaDriver.setExceptionsEnabled(true);
        cuInit(0);

        // Create a context
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);

        // Load the kernel
        CUmodule module1 = new CUmodule();
        cuModuleLoad(module1, ".\\src\\MergerHyperKernels.ptx");
        CUfunction mergerHelper1 = new CUfunction();
        cuModuleGetFunction(mergerHelper1, module1, "MergerHelper1");

        // create hollow blocks from hyperblocks
        ArrayList<HollowBlock> merging_hbs = new ArrayList<>();
        for (HyperBlock hyperBlock : hyper_blocks)
        {
            double[] mins = new double[DV.fieldLength];
            double[] maxes = new double[DV.fieldLength];

            for(int i = 0; i < hyperBlock.maximums.size(); i++){
                mins[i] = hyperBlock.minimums.get(i).get(0);
                maxes[i] = hyperBlock.maximums.get(i).get(0);
            }

            merging_hbs.add(new HollowBlock(Arrays.copyOf(maxes, maxes.length), Arrays.copyOf(mins, mins.length), hyperBlock.classNum));
        }

        hyper_blocks.clear();

        // create seed hollow blocks
        for (int i = 0; i < out_data.size(); i++)
        {
            for (int j = 0; j < out_data.get(i).size(); j++)
                merging_hbs.add(new HollowBlock(out_data.get(i).get(j), out_data.get(i).get(j), i));
        }

        boolean actionTaken = false;
        int cnt = merging_hbs.size();

        int time_cnt = 0;
        int total_time = 0;


        do
        {
            if (actionTaken || cnt < 1)
                cnt = merging_hbs.size();

            actionTaken = false;

            if (merging_hbs.isEmpty())
                break;

            int seedNum = -1;
            for (int i = 0; i < merging_hbs.size(); i++)
            {
                if (merging_hbs.get(i).mergable)
                {
                    seedNum = i;
                    break;
                }
            }

            if (seedNum == -1)
                break;

            HollowBlock seed_hb = merging_hbs.get(seedNum);
            merging_hbs.remove(seedNum);

            long startTime = System.currentTimeMillis();


            /***
             * MAKE PARALLEL
             */

            // Allocate device memory
            int numBlocks = merging_hbs.size();
            int fieldLength_dblBytes = DV.fieldLength * Double.BYTES;
            long hb_fieldLength_dblBytes = (long) numBlocks * fieldLength_dblBytes;

            CUdeviceptr seedHBMax = new CUdeviceptr();
            cuMemAlloc(seedHBMax, fieldLength_dblBytes);
            CUdeviceptr seedHBMin = new CUdeviceptr();
            cuMemAlloc(seedHBMin, fieldLength_dblBytes);

            CUdeviceptr mergingHBMaxes = new CUdeviceptr();
            cuMemAlloc(mergingHBMaxes, hb_fieldLength_dblBytes);
            CUdeviceptr mergingHBMins = new CUdeviceptr();
            cuMemAlloc(mergingHBMins, hb_fieldLength_dblBytes);

            CUdeviceptr combinedMax = new CUdeviceptr();
            cuMemAlloc(combinedMax, hb_fieldLength_dblBytes);
            CUdeviceptr combinedMin = new CUdeviceptr();
            cuMemAlloc(combinedMin, hb_fieldLength_dblBytes);

            int numOpClassPnts = 0;
            for (int i = 0; i < data.size(); i++)
            {
                if (seed_hb.classNum != i)
                    numOpClassPnts += data.get(i).size();
            }

            int[] host_toBeDeleted = new int[numBlocks];

            long numOpClassPnts_dblBytes = (long) numOpClassPnts * fieldLength_dblBytes;
            long numBlock_intBytes = (long) numBlocks * Integer.BYTES;

            CUdeviceptr oppClassPnts = new CUdeviceptr();
            cuMemAlloc(oppClassPnts, numOpClassPnts_dblBytes);
            CUdeviceptr toBeDeleted = new CUdeviceptr();
            cuMemAlloc(toBeDeleted, numBlock_intBytes);

            // Copy matrices A and B from host to device
            cuMemcpyHtoD(seedHBMax, Pointer.to(seed_hb.maximums), fieldLength_dblBytes);
            cuMemcpyHtoD(seedHBMin, Pointer.to(seed_hb.minimums), fieldLength_dblBytes);

            double[] host_mergingHBMaxes = new double[numBlocks * DV.fieldLength];
            double[] host_mergingHBMins = new double[numBlocks * DV.fieldLength];
            for (int i = 0; i < numBlocks; i++)
            {
                for (int j = 0; j < DV.fieldLength; j++)
                {
                    host_mergingHBMaxes[i * DV.fieldLength + j] = merging_hbs.get(i).maximums[j];
                    host_mergingHBMins[i * DV.fieldLength + j] = merging_hbs.get(i).minimums[j];
                }
            }

            cuMemcpyHtoD(mergingHBMaxes, Pointer.to(host_mergingHBMaxes), hb_fieldLength_dblBytes);
            cuMemcpyHtoD(mergingHBMins, Pointer.to(host_mergingHBMins), hb_fieldLength_dblBytes);

            double[] host_oppClassPnts = new double[numOpClassPnts * DV.fieldLength];
            for (int i = 0, k = 0; i < data.size(); i++)
            {
                if (seed_hb.classNum != i)
                {
                    for (int j = 0; j < data.get(i).size(); j++, k++)
                        System.arraycopy(data.get(i).get(j), 0, host_oppClassPnts, k * DV.fieldLength, DV.fieldLength);
                }
            }

            cuMemcpyHtoD(oppClassPnts, Pointer.to(host_oppClassPnts), numOpClassPnts_dblBytes);

            // Set up kernel parameters: pointers to matrix A, matrices B, result matrices, dimensions, and numMatrices
            Pointer kernelParameters = Pointer.to(
                    Pointer.to(seedHBMax),
                    Pointer.to(seedHBMin),
                    Pointer.to(mergingHBMaxes),
                    Pointer.to(mergingHBMins),
                    Pointer.to(combinedMax),
                    Pointer.to(combinedMin),
                    Pointer.to(oppClassPnts),
                    Pointer.to(toBeDeleted),
                    Pointer.to(new int[]{DV.fieldLength}),
                    Pointer.to(new int[]{numBlocks}),
                    Pointer.to(new int[]{numOpClassPnts * DV.fieldLength})
            );

            // Configure the kernel execution parameters
            int blockSize = 256;  // Thread block size
            int gridSize = merging_hbs.size() / blockSize + 1;

            // Launch the kernel
            cuLaunchKernel(mergerHelper1,
                    gridSize, 1, 1,    // Grid size (in blocks)
                    blockSize, 1, 1,  // Block size (in threads)
                    0, null,                    // Shared memory size and stream
                    kernelParameters, null      // Kernel parameters
            );
            cuCtxSynchronize();

            // Copy result matrices C from device to host
            cuMemcpyDtoH(Pointer.to(host_toBeDeleted), toBeDeleted, numBlock_intBytes);

            // wait for results then check if any action was taken
            for (int toDo : host_toBeDeleted)
            {
                if (toDo == 1)
                {
                    actionTaken = true;
                    break;
                }
            }

            // Clean up memory
            cuMemFree(seedHBMax);
            cuMemFree(seedHBMin);
            cuMemFree(mergingHBMaxes);
            cuMemFree(mergingHBMins);
            cuMemFree(combinedMax);
            cuMemFree(combinedMin);
            cuMemFree(oppClassPnts);
            cuMemFree(toBeDeleted);

            /***
             * MAKE PARALLEL
             */



            long endTime = System.currentTimeMillis();
            long executionTime = endTime - startTime;



            // delete hyperblocks in reverse
            // and add successfully combined hyperblocks
            ArrayList<HollowBlock> addHbs = new ArrayList<>();
            for (int i = host_toBeDeleted.length - 1; i >= 0; i--)
            {
                if (host_toBeDeleted[i] == 1)
                {
                    double[] maxPoint = new double[DV.fieldLength];
                    double[] minPoint = new double[DV.fieldLength];

                    // define combined space
                    for (int j = 0; j < DV.fieldLength; j++)
                    {
                        maxPoint[j] = Math.max(seed_hb.maximums[j], merging_hbs.get(i).maximums[j]);
                        minPoint[j] = Math.min(seed_hb.minimums[j], merging_hbs.get(i).minimums[j]);
                    }

                    HollowBlock hb = new HollowBlock(maxPoint, minPoint, seed_hb.classNum);

                    // remove old and add new
                    merging_hbs.remove(i);
                    addHbs.add(hb);
                }
            }

            merging_hbs.addAll(addHbs);

            if (!actionTaken)
            {
                seed_hb.mergable = false;
                merging_hbs.add(seed_hb);
            }

            //System.out.println("Cnt: " + cnt);
            //System.out.println("HB num: " + merging_hbs.size());
            //System.out.println("Seed HB Cls: " + seed_hb.classNum);
            //System.out.println("Check time: " + executionTime + " milliseconds");

            time_cnt++;
            total_time += executionTime;

            cnt--;

        } while (actionTaken || cnt > 0);


        //System.out.println("Average time: " + total_time / time_cnt + " milliseconds");

        //TODO:AUSTIN: MAKE SURE THIS IS A DEEP COPY
        // Create hyperblocks from merging blocks
        for (HollowBlock mergingHb : merging_hbs) {
            ArrayList<ArrayList<Double>> mins = new ArrayList<>();
            ArrayList<ArrayList<Double>> maxes = new ArrayList<>();

            for (int i = 0; i < mergingHb.minimums.length; i++) {
                // Convert each minimum and maximum into an ArrayList<Double>
                ArrayList<Double> minList = new ArrayList<>();
                ArrayList<Double> maxList = new ArrayList<>();

                minList.add(mergingHb.minimums[i]);
                maxList.add(mergingHb.maximums[i]);

                mins.add(minList);
                maxes.add(maxList);
            }

            // Add them to the list of blocks
            hyper_blocks.add(new HyperBlock(new ArrayList<>(maxes), new ArrayList<>(mins), mergingHb.classNum));
        }

        cuCtxDestroy(context);
    }

    private void merger_cuda_helper(int n, double[] seedHBMax, double[] seedHBMin, double[] mergingHBMaxes, double[] mergingHBMins, double[] combinedMax, double[] combinedMin, double[] opClassPnts, int[] toBeDeleted, int numDims, int numMergingHBs, int cases)
    {
        int offset = n * numDims;
        for (int i = 0; i < numDims; i++)
        {
            combinedMax[i+offset] = Math.max(seedHBMax[i], mergingHBMaxes[i+offset]);
            combinedMin[i+offset] = Math.min(seedHBMin[i], mergingHBMins[i+offset]);
        }

        // 1 = do merge, 0 = do not merge
        int merge = 1;
        for (int i = 0; i < cases; i += numDims)
        {
            boolean withinSpace = true;
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

        toBeDeleted[n] = merge;
    }




    //TODO: THIS WILL NEED A FULL REDO TO WORK WITH DISJUNCTIVE BLOCKS, BUT WE CAN USE THIS FOR NON-DISJUNCTIVE STILL
    public void saveHyperBlocksToCSV(String filePath)
    {
        try (FileWriter writer = new FileWriter(filePath))
        {
            // Write the header
            for (int i = 0; i < DV.fieldLength; i++)
                writer.append("Min  ").append(DV.fieldNames.get(i)).append(",");
            for (int i = 0; i < DV.fieldLength; i++)
                writer.append("Max  ").append(DV.fieldNames.get(i)).append(",");
            writer.append("Class\n");

            // Iterate through each HyperBlock
            for (HyperBlock hb : hyper_blocks)
            {
                ArrayList<ArrayList<Double>> min = hb.minimums;
                ArrayList<ArrayList<Double>> max = hb.maximums;
                int classNum = hb.classNum;


                // NOTE: ONLY IS PRINTING THE FIRST INTERVAL SET IF THERE IS AN OR IT WONT PRINT IT TO FILE RN.
                // Write the min values
                for (ArrayList<Double> value : min)
                    writer.append(String.valueOf(value.get(0))).append(",");

                // Write the max values
                for (ArrayList<Double> value : max)
                    writer.append(String.valueOf(value.get(0))).append(",");

                // Write the class
                writer.append(DV.uniqueClasses.get(classNum)).append("\n");
            }
        } catch (IOException e)
        {
            LOGGER.log(Level.SEVERE, e.toString(), e);
        }
    }


    public void saveDataObjectsToCSV(ArrayList<ArrayList<double[]>> dataObjects, String filePath)
    {
        try (FileWriter writer = new FileWriter(filePath))
        {
            // Write the header
            for (String fieldName : DV.fieldNames)
                writer.append(fieldName).append(",");

            writer.append("Class\n");

            // Iterate through each DataObject
            for (int i = 0; i < dataObjects.size(); i++)
            {
                ArrayList<double[]> data = dataObjects.get(i);
                for (double[] row : data)
                {
                    for (double value : row)
                        writer.append(String.valueOf(value)).append(",");

                    writer.append(DV.uniqueClasses.get(i)).append("\n");
                }
            }
        }
        catch (IOException e)
        {
            LOGGER.log(Level.SEVERE, e.toString(), e);
        }
    }





    /**
     * Gets all cases misclassified by the DV Linear Discriminant Function
     */
    private static void getMisclassifiedCases()
    {
        // create threads for each data object
        int numThreads = Math.min(Runtime.getRuntime().availableProcessors(), DV.trainData.size());
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        List<Future<Void>> futures = new ArrayList<>();

        // get misclassified cases
        DV.misclassifiedData = new ArrayList<>();
        for (int i = 0; i < DV.trainData.size(); i++)
        {
            DV.misclassifiedData.add(new ArrayList<>());
            if (i == DV.upperClass || DV.lowerClasses.get(i))
            {
                // execute task
                int finalI = i;
                Callable<Void> task = () -> misclassifiedCasesHelper(DV.trainData.get(finalI), finalI);
                Future<Void> future = executor.submit(task);
                futures.add(future);
            }
        }

        // get results from threads
        for (Future<Void> future : futures)
        {
            try
            {
                future.get();
            }
            catch (InterruptedException | ExecutionException e)
            {
                LOGGER.log(Level.SEVERE, e.toString(), e);
            }
        }

        executor.shutdown();
    }


    /**
     * Helper function for getMisclassifiedCases
     * @param data data used in classification
     * @param i class index
     */
    private static Void misclassifiedCasesHelper(DataObject data, int i)
    {
        for (int j = 0; j < data.coordinates.length; j++)
        {
            // check if endpoint is within the subset of used data
            double endpoint = data.coordinates[j][data.coordinates[j].length-1][0];
            if ((DV.domainArea[0] <= endpoint && endpoint <= DV.domainArea[1]) || !DV.domainActive)
            {
                // get classification
                if (i == DV.upperClass && DV.upperIsLower)
                {
                    // check if endpoint is correctly classified
                    if (endpoint > DV.threshold)
                        DV.misclassifiedData.get(i).add(DV.trainData.get(i).data[j]);
                }
                else if (i == DV.upperClass)
                {
                    // check if endpoint is correctly classified
                    if (endpoint < DV.threshold)
                        DV.misclassifiedData.get(i).add(DV.trainData.get(i).data[j]);
                }
                else if(DV.lowerClasses.get(i) && DV.upperIsLower)
                {
                    // check if endpoint is correctly classified
                    if (endpoint < DV.threshold)
                        DV.misclassifiedData.get(i).add(DV.trainData.get(i).data[j]);
                }
                else
                {
                    // check if endpoint is correctly classified
                    if (endpoint > DV.threshold)
                        DV.misclassifiedData.get(i).add(DV.trainData.get(i).data[j]);
                }
            }
        }

        return null;
    }
}
