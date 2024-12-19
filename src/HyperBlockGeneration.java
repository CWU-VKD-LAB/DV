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
import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.file.Files;
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

    // visualization options
    JRadioButton visualizeWithin;
    JRadioButton visualizeAll;
    JRadioButton visualizeOutline;
    JLabel graphLabel;
    JButton right;
    JButton left;
    int visualized_block = 0;

    // plot choice
    int plot_id = 0;
    
    // plot options
    JRadioButton PC;
    JRadioButton SPC;
    JRadioButton GLCL;
    JRadioButton PC_cross;
    JRadioButton SPC_less;
    JRadioButton GLCL_turn;
    JRadioButton GLCL_turn_dot;
    JRadioButton GLCL_x_proj_dot;

    // HB viewing options
    JRadioButton individualView;
    JRadioButton completeView;
    JRadioButton combinedClassView;
    JRadioButton combinedView;

    // colors
    Color[] graphColors;

    boolean remove_extra = false;


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
        // k-fold
        /*ProcessBuilder lda = new ProcessBuilder("cmd", "/c",
                "python",
                "source\\Python\\code_and_pyinstaller_spec\\kFoldSplit.py",
                "source\\Python\\train_data.csv",
                "source\\Python\\test_data.csv");

        try
        {
            // create file for python process
            CSV.createCSVDataObject(DV.trainData, "source\\Python\\train_data.csv");
            CSV.createCSVDataObject(DV.testData, "source\\Python\\test_data.csv");

            // run python (LDA) process
            Process process = lda.start();

            // read python outputs
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            String output;

            while ((output = reader.readLine()) != null)
            {
                // update angles
                System.out.println(output);
            }

            // delete created file
            File fileToDelete = new File("source\\Python\\train_data.csv");
            File fileToDelete2 = new File("source\\Python\\test_data.csv");
            Files.deleteIfExists(fileToDelete.toPath());
            Files.deleteIfExists(fileToDelete2.toPath());
        }
        catch (Exception e)
        {
            throw new RuntimeException(e);
        }*/

        test_HBs();


        // get average case
        /*for (int i = 0; i < average_case.size(); i++)
        {
            //for (int j = 0; j < average_case.get(0); j++)
            //{
                double[] input = new double[DV.fieldLength];
                //for (int j = 0; j < average_case.get(i).get(0).length; j++)
                    //input[j] = average_case.get(i).get(0)[j];// * (DataSetup.maxValues[j] - DataSetup.minValues[j]) + DataSetup.minValues[j];


                double[] best = DV.trainData.get(0).data[0];
                double best_dist = Double.MAX_VALUE;

                for (int d = 0; d < DV.trainData.size(); d++)
                {
                    for (int b = 0; b < DV.trainData.get(d).data.length; b++)
                    {
                        double new_dist = euclidean_distance(average_case.get(i).get(0), DV.trainData.get(d).data[b]);
                        if (new_dist < best_dist)
                        {
                            best = DV.trainData.get(d).data[b];
                            best_dist = euclidean_distance(average_case.get(i).get(0), DV.trainData.get(d).data[b]);
                        }
                    }
                }

                input = best;
                for (int j = 0; j < average_case.get(i).get(0).length; j++)
                    input[j] = input[j] * (DataSetup.maxValues[j] - DataSetup.minValues[j]) + DataSetup.minValues[j];


                // Create a BufferedImage
                BufferedImage image = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);

                // Fill the BufferedImage with the input data
                for (int y = 0; y < 28; y++) {
                    for (int x = 0; x < 28; x++) {
                        int index = y * 28 + x;
                        int value = (int) (input[index] * 255); // Assuming input values are normalized between 0 and 1
                        int rgb = (value << 16) | (value << 8) | value; // Convert to grayscale
                        image.setRGB(x, y, rgb);
                    }
                }

                // Save the BufferedImage to a file
                try {
                    ImageIO.write(image, "png", new File("D:\\Downloads\\image_" + DV.uniqueClasses.get(hyper_blocks.get(i).classNum) + ".png"));
                } catch (IOException e) {
                    e.printStackTrace();
                }
            //}
        }*/
    }


    HyperBlockGeneration(File filePath)
    {
        // set purity threshold
        if (explain_ldf)
            acc_threshold = DV.accuracy;

        // open file and read lines
        getData();
        hyper_blocks = DataSetup.setupHyperblockData(filePath);
        HB_analytics();

        // visualize hyperblocks
        if (!hyper_blocks.isEmpty())
        {
            HB_GUI();
            getPlot();
        }

        System.out.println("DONE!!!!");
    }

    /*
        Basically, we do nested for loop through the hyper_blocks.

        We will check if one block fully envelopes the other, if it doesn't we have to check case by

     */
    private void simplifyHBtoDisjunctiveForm(){
        Set<Integer> blocksToBeRemoved = new HashSet<>();
        // Go through each hyperblock
        for(int i = 0; i < hyper_blocks.size(); i++){
            HyperBlock outerBlock = hyper_blocks.get(i);
            for(int j = 0; j < hyper_blocks.size(); j++){
                // Don't compare a block to itself.
                if(i == j){continue;}

                if(blocksToBeRemoved.contains(i) || blocksToBeRemoved.contains(j)){continue;}

                // Get current inner block of same class
                HyperBlock innerBlock = hyper_blocks.get(j);
                if(outerBlock.classNum != innerBlock.classNum){
                    continue;
                }

                // First we want to check if the outer block is completely surrounding inner block. If it is,
                // we can remove the smaller block
                boolean allInside = true;
                for(int k = 0; k < DV.fieldLength; k++){
                    if(outerBlock.minimums.get(0)[k] > innerBlock.minimums.get(0)[k] || outerBlock.maximums.get(0)[k] < innerBlock.maximums.get(0)[k]){
                        allInside = false;
                        break;
                    }
                }

                if(allInside){
                    // This means we can delete the smaller block. (How will this effect our looping tho?)
                    blocksToBeRemoved.add(j);
                }else{
                    // Exhaustive search with updated intervals.
                    System.out.println("Doing an exhaustive search:" + i + "j: " + j);
                    ArrayList<double[]> mins = new ArrayList<>();
                    ArrayList<double[]> maxes = new ArrayList<>();

                    for(int k = 0; k < DV.fieldLength; k++){
                        if (outerBlock.maximums.get(0)[k] < innerBlock.minimums.get(0)[k] || innerBlock.maximums.get(0)[k] < outerBlock.minimums.get(0)[k]) {
                            // If not overlapping, we try to do an OR clause by taking both the inside and outside bounds. inside first.
                            maxes.add(k, new double[]{innerBlock.maximums.get(0)[k], outerBlock.maximums.get(0)[k]});
                            mins.add(k, new double[]{innerBlock.minimums.get(0)[k], outerBlock.minimums.get(0)[k]});
                        }else{
                            //TODO: THIS might battle with removeUselessAttributes. If so, make check to make sure not taking [0, 1]
                            // If they are overlapping take MAX(maxes) MIN(mins)
                            maxes.add(k, new double[]{(Math.max(outerBlock.maximums.get(0)[k], innerBlock.maximums.get(0)[k]))});
                            mins.add(k, new double[]{(Math.min(outerBlock.minimums.get(0)[k], innerBlock.minimums.get(0)[k]))});
                        }
                    }

                    // Go through all the data of other classes, check if it would fall into the new bounds of merged disjunctive block.
                    int classNum = outerBlock.classNum;
                    for(int k = 0; k < data.size(); k++) {
                        if(classNum == k){continue;} //Skip data of same class
                        boolean blockIsValid = true;

                        for (double[] point : data.get(k).data) {
                            System.out.println("Checking a point");
                            boolean pointInBounds = true;

                            for(int p_value = 0; p_value < point.length; p_value++){
                                boolean dimOutOfBounds = true;

                                // Make sure attribute is outside ALL OR's
                                for(int n = 0; n < maxes.get(p_value).length; n++){
                                    double min = mins.get(p_value)[n];
                                    double max = maxes.get(p_value)[n];

                                    // Check if point falls into the interval
                                    if(point[p_value] >= min && point[p_value] <= max){
                                        dimOutOfBounds = false;
                                        break;
                                    }
                                }

                                if(dimOutOfBounds){
                                    pointInBounds = false;
                                    break;
                                }
                            }

                            // If the point is in the bounds.
                            if(pointInBounds){
                                blockIsValid = false;
                                break;
                            }
                        }

                        if(blockIsValid){
                            //TODO: expand this to work with more than two options for the interval ranges
                            // Update the outer block to have the OR bounds
                            outerBlock.minimums = new ArrayList<>();
                            outerBlock.maximums = new ArrayList<>();

                            double[] mins1 = new double[DV.fieldLength];
                            double[] mins2 = new double[DV.fieldLength];
                            double[] maxes1 = new double[DV.fieldLength];
                            double[] maxes2 = new double[DV.fieldLength];

                            for(int m = 0; m < mins.size(); m++){
                               double[] mins_int = mins.get(m);
                               double[] maxes_int = maxes.get(m);

                               if(mins_int.length > 1){
                                   mins1[m] = mins_int[0];
                                   mins2[m] = mins_int[1];

                                   maxes1[m] = maxes_int[0];
                                   maxes2[m] = maxes_int[1];
                               }
                            }

                            outerBlock.minimums.add(Arrays.copyOf(mins1, mins1.length));
                            outerBlock.minimums.add(Arrays.copyOf(mins2, mins2.length));

                            outerBlock.maximums.add(Arrays.copyOf(maxes1, maxes1.length));
                            outerBlock.maximums.add(Arrays.copyOf(maxes2, maxes2.length));


                            // Delete the innerBlock.
                            blocksToBeRemoved.add(j);
                            System.out.println("Block: " + j + "Removed because of block: " + i);
                        }
                    }
                }
            }
        }

        List<Integer> sortedBlocksToBeRemoved = new ArrayList<>(blocksToBeRemoved);
        sortedBlocksToBeRemoved.sort(Collections.reverseOrder());
        for(int i : sortedBlocksToBeRemoved){
            hyper_blocks.remove(i);
        }

    }

    private void removeUselessAttributes()
    {

        // Go through all hyperblocks
        for(int i = 0; i < hyper_blocks.size(); i++){
            HyperBlock tempBlock = hyper_blocks.get(i);


            int classNum = tempBlock.classNum;
            double[] mins = Arrays.copyOf(tempBlock.minimums.get(0), tempBlock.minimums.get(0).length);
            double[] maxes = Arrays.copyOf(tempBlock.maximums.get(0), tempBlock.maximums.get(0).length);


            // removed represents one particular attribute which we want to try and remove
            for (int removed = 0; removed < maxes.length; removed++){

                //System.out.println("We are trying to remove attribute " + removed + " from hyperblock " + i + " which belongs to class " + tempBlock.classNum);
                boolean someoneInBounds = false;

                // iterating through all the OTHER classes, k representing a particular class number
                for(int k = 0; k < data.size(); k++){
                    // Skip going through data of same class as hyperblock

                    if(classNum == k){
                        continue;
                    }

                    // classNum of a hyperblock corresponds to index that data exists at in data. so data .get(classNum) to see all data from that class.
                    // Go through each point in the class

                    for(double[] point : data.get(k).data){
                        //System.out.println(Arrays.toString(point));

                        // check if the point is within ALL bounds of our hyperblock, except for the attribute we are looking at now
                        boolean inAllBounds = true;
                        for(int p_values = 0; p_values < point.length; p_values++){
                            // Skip the attribute we are trying to remove
                            if(p_values == removed){
                                continue;
                            }
                            // Check if p_value is less than max and greater than min for current range
                            if (point[p_values] < mins[p_values] || point[p_values] > maxes[p_values]){
                                inAllBounds = false;
                                break;
                            }
                        }
                        if (inAllBounds){
                            someoneInBounds = true;
                            break;
                        }
                    }

                }

                // if nobody falls into our bounds, we can safely remove now.
                if(!someoneInBounds){
                    //System.out.println("Removed : " + removed + " from: " + i);
                    // Update the maxes/mins to allow all range [0, 1] aka removing attribute
                    maxes[removed] = 1;
                    mins[removed] = 0;
                }
            }
            tempBlock.minimums.set(0, Arrays.copyOf(mins, mins.length));
            tempBlock.maximums.set(0, Arrays.copyOf(maxes, maxes.length));
        }
    }


    /**
     * Create hyperblocks using Interval Merger Hyper or Hyperblock Rules Linear
     */
    private void generateHBs(boolean remove_old)
    {
        try
        {
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

            long startTime = System.currentTimeMillis();

            while (!attributes.get(0).isEmpty())
            {
                // get largest hyperblock for each attribute
                ArrayList<DataATTR> intv = interval_hyper(executorService, attributes, acc_threshold, gen_hb);
                all_intv.add(intv);

                // if Hyperblock is unique then add
                if (intv.size() > 1)
                {
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

            // Create dataset without data from interval Hyperblocks
            ArrayList<ArrayList<double[]>> datum = new ArrayList<>();
            ArrayList<ArrayList<double[]>> seed_data = new ArrayList<>();
            ArrayList<ArrayList<Integer>> skips = new ArrayList<>();

            // all data
            for (DataObject dataObject : data)
            {
                datum.add(new ArrayList<>(Arrays.asList(dataObject.data)));
                seed_data.add(new ArrayList<>());
                skips.add(new ArrayList<>());
            }

            // find which data to skip
            for (ArrayList<DataATTR> dataATTRS : all_intv)
            {
                for (DataATTR dataATTR : dataATTRS)
                    skips.get(dataATTR.cl).add(dataATTR.cl_index);
            }

            for (ArrayList<Integer> skip : skips)
                Collections.sort(skip);

            for (int i = 0; i < data.size(); i++)
            {
                for (int j = 0; j < data.get(i).data.length; j++)
                {
                    if (!skips.get(i).isEmpty())
                    {
                        if (j != skips.get(i).get(0))
                            seed_data.get(i).add(data.get(i).data[j]);
                        else
                            skips.get(i).remove(0);
                    }
                    else
                        seed_data.get(i).add(data.get(i).data[j]);
                }
            }

            // sort data by most important attribute
            for (int i = 0; i < datum.size(); i++)
            {
                sortByColumn(datum.get(i), best_attribute);
                sortByColumn(seed_data.get(i), best_attribute);
            }

            // run dustin algorithm
            // create new threads equal to the number of processors
            //executorService = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
            //merger_hyperblock(executorService, datum, seed_data);
            merger_cuda(datum, seed_data);
            //executorService.shutdown();

            // order blocks from biggest to smallest by class
            order_hbs_by_class();
            // temp function removes all, but the largest HB in each class
            //order_hbs_by_size();
        }
        catch (InterruptedException | ExecutionException e)
        {
            LOGGER.log(Level.SEVERE, e.toString(), e);
            DV.warningPopup("Error Generating Hyperblocks", "Could not generate hyperblocks.");
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
            for (int j = 0; j < hb.hyper_block.size(); j++)
            {
                // if not unique, then return false
                if (!(intv_max < hb.minimums.get(j)[attr] || intv_min > hb.maximums.get(j)[attr]))
                {
                    return false;
                }
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
            for (int j = 0; j < hyperBlock.hyper_block.size(); j++)
                merging_hbs.add(new HollowBlock(hyperBlock.maximums.get(j), hyperBlock.minimums.get(j), hyperBlock.classNum));
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
            for (HollowBlock mergingHb : merging_hbs)
                hyper_blocks.add(new HyperBlock(mergingHb.maximums, mergingHb.minimums, mergingHb.classNum));
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

        mainFrame.setMinimumSize(new Dimension(DV.minSize[0], DV.minSize[1]));

        // show
        mainFrame.setVisible(true);
        mainFrame.revalidate();
        mainFrame.pack();
        mainFrame.repaint();
    }


    /**
     * Create a toolbar for the hyperblock GUI
     * @return toolbar
     */
    private JToolBar createToolBar()
    {
        JToolBar toolBar = new JToolBar();
        toolBar.setFloatable(false);
        toolBar.setRollover(true);

        // GLC-L plot options
        JLabel plots = new JLabel("Plot Options: ");
        ButtonGroup plotBtns = new ButtonGroup();
        plots.setFont(plots.getFont().deriveFont(Font.BOLD, 12f));
        toolBar.add(plots);
        toolBar.addSeparator();

        PC = new JRadioButton("PC", true);
        PC.addActionListener(pce ->
        {
            plot_id = 0;
            updateGraphs();
        });
        toolBar.add(PC);
        plotBtns.add(PC);
        toolBar.addSeparator();

        GLCL = new JRadioButton("GLC-L", false);
        GLCL.addActionListener(pce ->
        {
            plot_id = 1;
            updateGraphs();
        });
        toolBar.add(GLCL);
        plotBtns.add(GLCL);
        toolBar.addSeparator();

        SPC = new JRadioButton("SPC-1B", false);
        SPC.addActionListener(pce ->
        {
            plot_id = 2;
            updateGraphs();
        });
        toolBar.add(SPC);
        plotBtns.add(SPC);
        toolBar.addSeparator();

        PC_cross = new JRadioButton("PC-2n", false);
        PC_cross.addActionListener(pcce ->
        {
            plot_id = 3;
            updateGraphs();
        });
        toolBar.add(PC_cross);
        plotBtns.add(PC_cross);
        toolBar.addSeparator();

        SPC_less = new JRadioButton("SPC-2B", false);
        SPC_less.addActionListener(spcle ->
        {
            plot_id = 4;
            updateGraphs();
        });
        toolBar.add(SPC_less);
        plotBtns.add(SPC_less);
        toolBar.addSeparator();
        toolBar.addSeparator();
        toolBar.addSeparator();

        JLabel views = new JLabel("View Options: ");
        ButtonGroup viewBtns = new ButtonGroup();
        views.setFont(views.getFont().deriveFont(Font.BOLD, 12f));
        toolBar.add(views);
        toolBar.addSeparator();

        individualView = new JRadioButton("Individual Hyperblocks", true);
        individualView.setToolTipText("View all hyperblocks separately.");
        individualView.addActionListener(e ->
        {
            // set left and right buttons
            navPanel.removeAll();
            navPanel.add(left);
            navPanel.add(right);

            if (visualized_block > hyper_blocks.size() - 1)
                visualized_block = 0;

            getPlot();
        });
        toolBar.add(individualView);
        viewBtns.add(individualView);
        toolBar.addSeparator();

        completeView = new JRadioButton("All Blocks");
        completeView.setToolTipText("View all hyperblocks.");
        completeView.addActionListener(e ->
        {
            // remove left and right buttons
            navPanel.removeAll();
            getPlot();
        });
        toolBar.add(completeView);
        viewBtns.add(completeView);
        toolBar.addSeparator();

        combinedClassView = new JRadioButton("Class Combined Blocks");
        combinedClassView.setToolTipText("View all hyperblocks with hyperblocks of the same class combined.");
        combinedClassView.addActionListener(e ->
        {
            navPanel.removeAll();
            getPlot();
        });
        toolBar.add(combinedClassView);
        viewBtns.add(combinedClassView);
        toolBar.addSeparator();

        combinedView = new JRadioButton("Combined Blocks");
        combinedView.setToolTipText("View all hyperblocks combined.");
        combinedView.addActionListener(e ->
        {
            navPanel.removeAll();
            getPlot();
        });
        toolBar.add(combinedView);
        viewBtns.add(combinedView);
        toolBar.addSeparator();
        toolBar.addSeparator();
        toolBar.addSeparator();

        JLabel blockType = new JLabel("Hyperblock Options: ");
        ButtonGroup dataBtns = new ButtonGroup();
        blockType.setFont(blockType.getFont().deriveFont(Font.BOLD, 12f));
        toolBar.add(blockType);
        toolBar.addSeparator();

        // default blocks
        JRadioButton defaultBlocksBtn = new JRadioButton("Default Blocks", true);
        defaultBlocksBtn.setToolTipText("Rebuild blocks using all data.");
        defaultBlocksBtn.addActionListener(e ->
        {
            getData();
            updateGraphs();
        });
        toolBar.add(defaultBlocksBtn);
        dataBtns.add(defaultBlocksBtn);
        toolBar.addSeparator();

        // overlap blocks
        JRadioButton overlapBlocksBtn = new JRadioButton("Overlap Blocks");
        overlapBlocksBtn.setToolTipText("Rebuild blocks using only data in the overlap area.");
        overlapBlocksBtn.addActionListener(e ->
        {
            getOverlapData();
            updateGraphs();
        });
        toolBar.add(overlapBlocksBtn);
        dataBtns.add(overlapBlocksBtn);
        toolBar.addSeparator();

        // non-overlap blocks
        JRadioButton nonOverlapBlocksBtn = new JRadioButton("Non-Overlap Blocks");
        nonOverlapBlocksBtn.setToolTipText("Rebuild blocks using only data not in the overlap area.");
        nonOverlapBlocksBtn.addActionListener(e ->
        {
            getNonOverlapData();
            updateGraphs();
        });
        toolBar.add(nonOverlapBlocksBtn);
        dataBtns.add(nonOverlapBlocksBtn);
        toolBar.addSeparator();
        toolBar.addSeparator();
        toolBar.addSeparator();

        JLabel dataView = new JLabel("Data View Options: ");
        ButtonGroup dataVisBtns = new ButtonGroup();
        dataView.setFont(dataView.getFont().deriveFont(Font.BOLD, 12f));
        toolBar.add(dataView);
        toolBar.addSeparator();

        visualizeWithin = new JRadioButton("Within Block", true);
        visualizeWithin.setToolTipText("Only Visualize Datapoints Within Block");
        visualizeWithin.addActionListener(al -> updateGraphs());
        toolBar.add(visualizeWithin);
        dataVisBtns.add(visualizeWithin);
        toolBar.addSeparator();

        visualizeAll = new JRadioButton("All Data", false);
        visualizeAll.setToolTipText("Visualize All Datapoints");
        visualizeAll.addActionListener(al -> updateGraphs());
        toolBar.add(visualizeAll);
        dataVisBtns.add(visualizeAll);
        toolBar.addSeparator();

        visualizeOutline = new JRadioButton("Outline Block", false);
        visualizeOutline.setToolTipText("Only Visualize Outline of Block");
        visualizeOutline.addActionListener(al -> updateGraphs());
        toolBar.add(visualizeOutline);
        dataVisBtns.add(visualizeOutline);
        toolBar.addSeparator();
        toolBar.addSeparator();
        toolBar.addSeparator();

        // Simplify by removing attributes
        JButton removeUselessBtn = new JButton("Simplify Blocks");
        toolBar.add(removeUselessBtn);
        removeUselessBtn.addActionListener(e -> {
            removeUselessAttributes();
            HB_analytics();
            updateGraphs();
        });

        // HB level
        JLabel lvlView = new JLabel("HB Level: ");
        lvlView.setFont(lvlView.getFont().deriveFont(Font.BOLD, 12f));
        toolBar.add(lvlView);

        hb_lvl = new JSpinner(new SpinnerNumberModel(1, 1, 3, 1));
        hb_lvl.setEditor(new JSpinner.NumberEditor(hb_lvl, "0"));
        hb_lvl.addChangeListener(hbe ->
        {
            if ((Integer) hb_lvl.getValue() == 2 || (Integer) hb_lvl.getValue() == 3)
            {
                increase_level();
                updateGraphs();
            }
        });
        toolBar.add(hb_lvl);
        toolBar.addSeparator();

        // test HBs
        JButton HB_test = new JButton("Test HBs");
        HB_test.addActionListener(e -> test_HBs());
        //toolBar.add(HB_test);

        JButton HB_extra = new JButton("Remove Extra");
        HB_extra.addActionListener(e ->
        {
            remove_extra = !remove_extra;
            updateGraphs();
        });
        //toolBar.add(HB_extra);

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

        // get new visuilization
        switch (plot_id)
        {
            case 0 ->
            { // PC
                if (individualView.isSelected())
                {
                    graphPanel.add(PC_HB(data, visualized_block));
                }
                else if (completeView.isSelected())
                {
                    graphPanel.add(PC_HBs(data));
                }
                else if (combinedClassView.isSelected())
                {
                }
                else if (combinedView.isSelected())
                {
                }
            }
            case 1 ->
            { // GLC-L
                if (individualView.isSelected())
                {
                    graphPanel.add(GLC_L_HB(data, visualized_block));
                }
                else if (completeView.isSelected())
                {
                }
                else if (combinedClassView.isSelected())
                {
                }
                else if (combinedView.isSelected())
                {
                }
            }
            case 2 ->
            { // SPC
                if (individualView.isSelected())
                {
                }
                else if (completeView.isSelected())
                {
                }
                else if (combinedClassView.isSelected())
                {
                }
                else if (combinedView.isSelected())
                {
                }
            }
            case 3 ->
            { // PC cross
                if (individualView.isSelected())
                {
                }
                else if (completeView.isSelected())
                {
                }
                else if (combinedClassView.isSelected())
                {
                }
                else if (combinedView.isSelected())
                {
                }
            }
            case 4 ->
            { // SPC reduced
                if (individualView.isSelected())
                {
                }
                else if (completeView.isSelected())
                {
                }
                else if (combinedClassView.isSelected())
                {
                }
                else if (combinedView.isSelected())
                {
                }
            }
        }

        // show new visualization
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


                // Go through all allowed intervals for the attribute
                for(int j = 0; j < tempB.maximums.get(i).size(); j++){
                    if(tempB.maximums.get(i).get(j) > 0){
                        // If multiple intervals appear
                        if(j > 0){
                            rule.append(" OR ");
                        }

                        rule.append(String.format("%.2f &le; X%d &le; %.2f", tempB.minimums.get(i).get(j), i, tempB.maximums.get(i).get(j)));

                        if (i != DV.fieldLength - 1)
                            rule.append(", ");
                        else
                            rule.append(", then class ").append(DV.uniqueClasses.get(tempB.classNum));
                    }
                }
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

    private void updateGraphs()
    {
        if (individualView.isSelected())
            individualView.doClick();
        else if (completeView.isSelected())
            completeView.doClick();
        else if (combinedClassView.isSelected())
            combinedClassView.doClick();
        else
            combinedView.doClick();
    }


    //TODO:AUSTIN: Adjusted to NOT BREAK with new mins/max format, but needs to be reworked to work and display the whole block.
    /**
     * MAYBE ABSTRACT LOOPS TO ADD DATA LATER
     */

    /**
     * Create a PC visualization for a single hyper-block
     * @param datum data to visualize
     * @param visualized_block hyper-block to visualize
     * @return panel with PC visualization
     */
    private ChartPanel PC_HB(ArrayList<DataObject> datum, int visualized_block)
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

        int lineCnt = 0;
        for (int d = 0; d < datum.size(); d++)
        {
            for (int i = 0; i < datum.get(d).data.length; i++)
            {
                // start line at (0, 0)
                XYSeries line = new XYSeries(lineCnt, false, true);
                boolean within = false;
                int within_block = 0;

                for (int k = 0; k < tempBlock.hyper_block.size(); k++)
                {
                    if (inside_HB(visualized_block, k, datum.get(d).data[i]))
                    {
                        within_block = k;
                        within = true;
                    }
                }

                // add points to lines
                int off = 0;
                for (int j = 0; j < DV.fieldLength; j++)
                    if (remove_extra && tempBlock.minimums.get(j).get(0) != 0.5 && tempBlock.maximums.get(j).get(0) != 0.5)
                    {
                        line.add(off, datum.get(d).data[i][j]);
                        off++;
                    }
                    else if (!remove_extra)
                        line.add(j, datum.get(d).data[i][j]);

                // add endpoint and timeline
                if (visualizeWithin.isSelected() && within)
                {
                    // add series
                    if (d == tempBlock.classNum)
                    {
                        goodGraphLines.addSeries(line);
                        goodLineRenderer.setSeriesPaint(lineCnt, graphColors[d]);
                        goodLineRenderer.setSeriesStroke(lineCnt, strokes[within_block]);
                    }
                    else
                    {
                        badGraphLines.addSeries(line);
                        badLineRenderer.setSeriesPaint(lineCnt, graphColors[d]);
                        badLineRenderer.setSeriesStroke(lineCnt, strokes[within_block]);
                    }

                    lineCnt++;
                }
                else if (visualizeAll.isSelected())
                {
                    // add series
                    goodGraphLines.addSeries(line);
                    goodLineRenderer.setSeriesPaint(lineCnt, graphColors[d]);
                    goodLineRenderer.setSeriesStroke(lineCnt, strokes[within_block]);
                    lineCnt++;
                }
            }
        }

        // populate average series
        for (int k = 0; k < average_case.get(visualized_block).size(); k++)
        {
            if (tempBlock.hyper_block.get(k).size() > 1)
            {
                XYSeries line = new XYSeries(k, false, true);
                int off = 0;
                for (int i = 0; i < average_case.get(visualized_block).get(k).length; i++)
                    if (remove_extra && tempBlock.minimums.get(i).get(0) != 0.5 && tempBlock.maximums.get(i).get(0) != 0.5)
                    {
                        line.add(off, average_case.get(visualized_block).get(k)[i]);
                        off++;
                    }
                    else if (!remove_extra)
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
                int cnt = 0;
                for (int j = 0; j < DV.fieldLength; j++)
                {
                    if (remove_extra && tempBlock.minimums.get(j).get(0) != 0.5 && tempBlock.maximums.get(j).get(0) != 0.5)
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
                    if (remove_extra && tempBlock.minimums.get(j).get(0) != 0.5 && tempBlock.maximums.get(j).get(0) != 0.5)
                    {
                        tmp1.add(cnt, tempBlock.maximums.get(j).get(0));
                        tmp2.add(cnt, tempBlock.maximums.get(j).get(0));
                        cnt--;
                    }
                    else if (!remove_extra)
                    {
                        tmp1.add(j, tempBlock.maximums.get(j).get(0));
                        tmp2.add(j, tempBlock.maximums.get(j).get(0));
                    }
                }

                // Hyperblock is always size 1, so k is always 0
                // this will grab the first element in HB mins
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



    private JScrollPane PC_HBs(ArrayList<DataObject> data)
    {
        // create panel with 3 columns
        JPanel tilePanel = new JPanel();
        tilePanel.setLayout(new GridLayout((int)Math.ceil(hyper_blocks.size() / 3.0), 3));
        JScrollPane tileScroll = new JScrollPane(tilePanel);

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
                for (int k = 0; k < hyper_blocks.get(visualized_block).hyper_block.size(); k++)
                {
                    if (inside_HB(visualized_block, k, data.data[i]))
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
                for (int k = 0; k < hyper_blocks.get(visualized_block).hyper_block.size(); k++)
                {
                    if (inside_HB(visualized_block, k, data.data[i]))
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
            if (hyper_blocks.get(visualized_block).hyper_block.get(k).size() > 1)
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
        for (int k = 0, offset = 0; k < hyper_blocks.get(visualized_block).hyper_block.size(); k++)
        {
            if (hyper_blocks.get(visualized_block).hyper_block.get(k).size() > 1)
            {
                XYSeries tmp1 = new XYSeries(k-offset, false, true);
                XYSeries tmp2 = new XYSeries(k-offset, false, true);
                int cnt = 0;
                for (int j = 0; j < DV.fieldLength; j++)
                {
                    if (remove_extra && Math.abs(average_case.get(0).get(0)[j] - average_case.get(1).get(0)[j]) > 0.25)
                    {
                        tmp1.add(cnt, hyper_blocks.get(visualized_block).minimums.get(k)[j]);
                        tmp2.add(cnt, hyper_blocks.get(visualized_block).minimums.get(k)[j]);
                        cnt++;
                    }
                    else if (!remove_extra)
                    {
                        tmp1.add(j, hyper_blocks.get(visualized_block).minimums.get(k)[j]);
                        tmp2.add(j, hyper_blocks.get(visualized_block).minimums.get(k)[j]);
                    }
                }

                for (int j = DV.fieldLength - 1; j > -1; j--)
                {
                    if (remove_extra && Math.abs(average_case.get(0).get(0)[j] - average_case.get(1).get(0)[j]) > 0.25)
                    {
                        tmp1.add(cnt, hyper_blocks.get(visualized_block).maximums.get(k)[j]);
                        tmp2.add(cnt, hyper_blocks.get(visualized_block).maximums.get(k)[j]);
                        cnt--;
                    }
                    else if (!remove_extra)
                    {
                        tmp1.add(j, hyper_blocks.get(visualized_block).maximums.get(k)[j]);
                        tmp2.add(j, hyper_blocks.get(visualized_block).maximums.get(k)[j]);
                    }
                }

                tmp1.add(0, hyper_blocks.get(visualized_block).minimums.get(k)[0]);
                tmp2.add(0, hyper_blocks.get(visualized_block).minimums.get(k)[0]);

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
        XYPlot plot = ChartsAndPlots.createHBPlot(pcChart, graphColors[hyper_blocks.get(visualized_block).classNum]);
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
                boolean within = false;
                for (int k = 0; k < hyper_blocks.get(visualized_block).hyper_block.size(); k++)
                {
                    if (inside_HB(visualized_block, k, data.data[i]))
                        within = true;
                }

                if (within)
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
                for (int k = 0; k < hyper_blocks.get(visualized_block).hyper_block.size(); k++)
                {
                    if (inside_HB(visualized_block, k, data.data[i]))
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
                    goodLineRenderer.setSeriesPaint(lineCnt, graphColors[hyper_blocks.get(visualized_block).classNum]);//lineClrs[lineCnt]);
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
            if (hyper_blocks.get(visualized_block).hyper_block.get(k).size() > 1)
            {
                XYSeries line = new XYSeries(k, false, true);
                for (int i = 0; i < average_case.get(visualized_block).get(k).length; i++)
                    if (remove_extra && hyper_blocks.get(visualized_block).minimums.get(0)[i] != hyper_blocks.get(visualized_block).maximums.get(0)[i])
                        line.add(i, average_case.get(visualized_block).get(k)[i]);

                avgLines.addSeries(line);
                avgRenderer.setSeriesPaint(k, Color.BLACK);
            }
        }

        // add hyperblocks
        for (int k = 0, offset = 0; k < hyper_blocks.get(visualized_block).hyper_block.size(); k++)
        {
            if (hyper_blocks.get(visualized_block).hyper_block.get(k).size() > 1)
            {
                XYSeries tmp1 = new XYSeries(k-offset, false, true);
                XYSeries tmp2 = new XYSeries(k-offset, false, true);

                for (int j = 0; j < DV.fieldLength; j++)
                {
                    if (remove_extra && hyper_blocks.get(visualized_block).minimums.get(0)[j] != hyper_blocks.get(visualized_block).maximums.get(0)[j])
                    {
                        tmp1.add(j, hyper_blocks.get(visualized_block).minimums.get(k)[j]);
                        tmp2.add(j, hyper_blocks.get(visualized_block).minimums.get(k)[j]);
                    }
                }

                for (int j = DV.fieldLength - 1; j > -1; j--)
                {
                    if (remove_extra && hyper_blocks.get(visualized_block).minimums.get(0)[j] != hyper_blocks.get(visualized_block).maximums.get(0)[j])
                    {
                        tmp1.add(j, hyper_blocks.get(visualized_block).maximums.get(k)[j]);
                        tmp2.add(j, hyper_blocks.get(visualized_block).maximums.get(k)[j]);
                    }
                }

                for (int j = 0; j < DV.fieldLength; j++)
                {
                    if (remove_extra && hyper_blocks.get(visualized_block).minimums.get(0)[j] != hyper_blocks.get(visualized_block).maximums.get(0)[j])
                    {
                        tmp1.add(j, hyper_blocks.get(visualized_block).maximums.get(k)[j]);
                        tmp2.add(j, hyper_blocks.get(visualized_block).maximums.get(k)[j]);
                        break;
                    }
                }
                //tmp1.add(0, hyper_blocks.get(visualized_block).minimums.get(k)[0]);
                //tmp2.add(0, hyper_blocks.get(visualized_block).minimums.get(k)[0]);

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
        XYPlot plot = ChartsAndPlots.createPlot(pcChart, hyper_blocks.get(visualized_block).classNum);
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
        ArrayList<double[]> cases = new ArrayList<>();

        boolean[] max = new boolean[DV.fieldLength];
        boolean[] min = new boolean[DV.fieldLength];

        for (int i = 0; i < DV.fieldLength; i++)
        {
            max[i] = true;
            min[i] = true;
        }

        for (int i = 0; i < hyper_blocks.get(num).hyper_block.get(0).size(); i++)
        {
            boolean part = false;

            for (int j = 0; j < DV.fieldLength; j++)
            {
                if (hyper_blocks.get(num).hyper_block.get(0).get(i)[j] == hyper_blocks.get(num).maximums.get(0)[j] && max[j])
                {
                    part = true;
                    max[j] = false;
                }
                else if (hyper_blocks.get(num).hyper_block.get(0).get(i)[j] == hyper_blocks.get(num).minimums.get(0)[j] && min[j])
                {
                    part = true;
                    min[j] = false;
                }
            }

            if (part)
                cases.add(hyper_blocks.get(num).hyper_block.get(0).get(i));
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


    /**
     * Orders hyperblocks by size
     */
    private void order_hbs_by_size()
    {
        //hyper_blocks.sort(new HBComparator());


        ArrayList<ArrayList<HyperBlock>> ordered = new ArrayList<>();
        for (int i = 0; i < DV.classNumber; i++)
            ordered.add(new ArrayList<>());

        for (HyperBlock hyperBlock : hyper_blocks)
            ordered.get(hyperBlock.classNum).add(hyperBlock);

        hyper_blocks.clear();
        for (ArrayList<HyperBlock> hyperBlocks : ordered)
        {
            hyperBlocks.sort(new HBComparator());
            hyper_blocks.add(hyperBlocks.get(0));
        }
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
            for (int i = 0; i < DV.trainData.size(); i++)
            {
                for (int j = 0; j < DV.trainData.get(i).data.length; j++)
                {
                    for (int q = 0; q < hyper_blocks.get(h).hyper_block.size(); q++)
                    {
                        if (inside_HB(h, q, DV.trainData.get(i).data[j]))
                        {
                            if (i == hyper_blocks.get(h).classNum)
                            {
                                maj_cnt++;
                                global_maj++;
                            }


                            counter[h]++;
                            global_cnt++;
                            hyper_blocks.get(h).hyper_block.get(q).add(DV.trainData.get(i).data[j]);
                        }
                    }
                }
            }

            System.out.println("\nBlock " + (h+1) + " Size: " + counter[h]);
            System.out.println("Block " + (h+1) + " Accuracy: " + (maj_cnt / counter[h]));

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

        System.out.println("TOTAL NUM IN BLOCKS: " + bcnt);

        for (int i = 0; i < DV.trainData.size(); i++)
        {
            for (int j = 0; j < DV.trainData.get(i).data.length; j++)
            {
                boolean in = false;
                for (int h = 0; h < hyper_blocks.size(); h++)
                {
                    for (int q = 0; q < hyper_blocks.get(h).hyper_block.size(); q++)
                    {
                        if (inside_HB(h, q, DV.trainData.get(i).data[j]))
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

        System.out.println("NOT IN ANY BLOCKS: " + not_in + "\n");
        System.out.println("Total Accuracy: " + (global_maj / global_cnt));
    }


    /**
     * Checks if data is inside a hyperblock
     * @param hb1 Hyperblock number
     * @param hb2 Hyperblock sub-block number
     * @param data Data to check
     * @return True if data is inside hyperblock
     */
    private boolean inside_HB(int hb1, int hb2, double[] data)
    {
        boolean inside = true;
        for (int j = 0; j < DV.fieldLength; j++)
        {
            if (data[j] > hyper_blocks.get(hb1).maximums.get(hb2)[j] ||
                    data[j] < hyper_blocks.get(hb1).minimums.get(hb2)[j])
            {
                inside = false;
                break;
            }
        }

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

                for (HyperBlock hyperBlock : hyper_blocks)
                {
                    for (int q = 0; q < hyperBlock.hyper_block.size(); q++)
                    {
                        boolean inside = true;
                        for (int k = 0; k < DV.fieldLength; k++)
                        {
                            if (DV.testData.get(i).data[j][k] > hyperBlock.maximums.get(q)[k] ||
                                    DV.testData.get(i).data[j][k] < hyperBlock.minimums.get(q)[k])
                            {
                                inside = false;
                                break;
                            }
                        }

                        if (inside)
                        {
                            classified = true;

                            if (i == hyperBlock.classNum)
                                good++;
                            else
                            {
                                bad++;
                                bad_cases.add(DV.testData.get(i).data[j]);
                                bad_cases_classes.add(i);
                            }
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
                    double newLocalMax = Math.max(tmp.maximums.get(0)[j], blocks.get(i).maximums.get(0)[j]);
                    double newLocalMin = Math.min(tmp.minimums.get(0)[j], blocks.get(i).minimums.get(0)[j]);

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
                    double newLocalMax = Math.max(tmp.maximums.get(0)[j], block.maximums.get(0)[j]);
                    double newLocalMin = Math.min(tmp.minimums.get(0)[j], block.minimums.get(0)[j]);

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
                    double newLocalMax = Math.max(tmp.maximums.get(0)[j], blocks.get(highestAccIndex).maximums.get(0)[j]);
                    double newLocalMin = Math.min(tmp.minimums.get(0)[j], blocks.get(highestAccIndex).minimums.get(0)[j]);

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
        //TODO: Change this to not requre the user to change the code
        cuModuleLoad(module1, ".\\src\\MergerHyperKernels.ptx");
        CUfunction mergerHelper1 = new CUfunction();
        cuModuleGetFunction(mergerHelper1, module1, "MergerHelper1");

        /*CUmodule module2 = new CUmodule();
        cuModuleLoad(module2, "D:\\GitHub\\DV\\src\\MergerHyperKernels.ptx");
        CUfunction mergerHelper2 = new CUfunction();
        cuModuleGetFunction(mergerHelper2, module2, "MergerHelper2");*/


        // create hollow blocks from hyperblocks
        ArrayList<HollowBlock> merging_hbs = new ArrayList<>();
        for (HyperBlock hyperBlock : hyper_blocks)
        {
            for (int j = 0; j < hyperBlock.hyper_block.size(); j++)
                merging_hbs.add(new HollowBlock(hyperBlock.maximums.get(j), hyperBlock.minimums.get(j), hyperBlock.classNum));
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

            System.out.println("Cnt: " + cnt);
            System.out.println("HB num: " + merging_hbs.size());
            System.out.println("Seed HB Cls: " + seed_hb.classNum);
            System.out.println("Check time: " + executionTime + " milliseconds");

            time_cnt++;
            total_time += executionTime;

            cnt--;

        } while (actionTaken || cnt > 0);



        System.out.println("Average time: " + total_time / time_cnt + " milliseconds");



        // create hyperblocks from mergine blocks
        for (HollowBlock mergingHb : merging_hbs)
            hyper_blocks.add(new HyperBlock(mergingHb.maximums, mergingHb.minimums, mergingHb.classNum));

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
                double[] min = hb.minimums.get(0);
                double[] max = hb.maximums.get(0);
                int classNum = hb.classNum;

                // Write the min values
                for (double value : min)
                    writer.append(String.valueOf(value)).append(",");

                // Write the max values
                for (double value : max)
                    writer.append(String.valueOf(value)).append(",");

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
