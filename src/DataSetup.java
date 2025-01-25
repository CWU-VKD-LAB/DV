import javax.swing.*;
import java.awt.*;
import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.List;
import java.util.concurrent.*;

public class DataSetup
{
    // classes for data
    static ArrayList<String> allClasses;

    // min/max for each feature
    static double[] maxValues;
    static double[] minValues;

    // mean/sd for each feature
    static double[] mean;
    static double[] sd;

    // default all columns
    static boolean setValues = false;


    /**
     * Sets up DV with data from datafile
     * @param dataFile .csv file holding data
     * @return whether setupWithData was successful
     */
    public static boolean setupWithData(File dataFile)
    {
        // data string[][] representation of dataFile (csv)
        String[][] stringData = getStringFromCSV(dataFile);
        if (stringData != null)
        {
            // prepare data, classes, and attributes
            stringData = formatData(stringData);
            setupClassesAndAttributes(stringData);

            // get numerical data from string data
            double[][] numericalData = stringToNumerical(stringData);
            if (numericalData != null)
                return setupData(numericalData);
            else
                return false;
        }
        else
            return false;
    }


    /**
     * Sets up DV with data from datafile
     * @return whether setupWithData was successful
     */
    public static boolean setupWithData(String[][] stringData)
    {
        // prepare data, classes, and attributes
        stringData = formatData(stringData);
        setupClassesAndAttributes(stringData);

        // get numerical data from string data
        double[][] numericalData = stringToNumerical(stringData);
        if (numericalData != null)
            return setupData(numericalData);
        else
            return false;
    }


    /**
     * Purges classes and ID from stringData
     * @param stringData String[][] representation of input data
     * @return String[][] representation of input data without classes and ID
     */
    private static String[][] formatData(String[][] stringData)
    {
        // get classes and class number then removes classes from dataset
        if (DV.hasClasses)
        {
            allClasses = getClasses(stringData);
            stringData = purgeClasses(stringData);
        }
        else
        {
            DV.classNumber = 1;
            DV.uniqueClasses = new ArrayList<>(List.of("N/A"));
        }

        // remove ID
        if (DV.hasID)
            stringData = purgeID(stringData);

        return stringData;
    }


    /**
     * Instantiates classes, attributes, and angles for DV
     * @param stringData String[][] representation of input data
     */
    private static void setupClassesAndAttributes(String[][] stringData)
    {
        if (!DV.classification)
            DV.classNumber = 1;

        // set class visualization -> all classes except the first go on the lower graph
        DV.upperClass = 0;
        DV.lowerClasses = new ArrayList<>(List.of(false));
        for (int i = 1; i < DV.classNumber; i++)
            DV.lowerClasses.add(true);

        // get fieldNames and fieldLength
        DV.fieldNames = getFieldNames(stringData);
        DV.fieldLength = DV.fieldNames.size();
        DV.standardFieldNames = new ArrayList<>(DV.fieldNames);
        DV.standardFieldLength = DV.fieldLength;

        // initialize original attribute order
        DV.originalAttributeOrder = new ArrayList<>();
        for (int i = 0; i < DV.fieldLength; i++)
            DV.originalAttributeOrder.add(i);

        // initializes angles to 45 degrees
        DV.angles = new double[DV.fieldLength];
        DV.prevAngles = new double[DV.fieldLength];
        DV.standardAngles = new double[DV.fieldLength];
        DV.activeAttributes = new ArrayList<>();
        for (int i = 0; i < DV.fieldLength; i++)
        {
            DV.angles[i] = 45;
            DV.prevAngles[i] = 45;
            DV.activeAttributes.add(true);
        }
    }


    /**
     * Creates DataObjects from numerical data
     * @param numericalData numerical data
     * @return DataObjects
     */
    private static boolean setupData(double[][] numericalData)
    {
        // save original data
        double[][] originalNumericalData = new double[numericalData.length][];
        for (int i = 0; i < numericalData.length; i++)
            originalNumericalData[i] = Arrays.copyOf(numericalData[i], DV.fieldLength);

        // normalize data
        double[][] normalizedNumericalData = normalizeData(numericalData);

        if (DV.hasClasses)
        {
            ArrayList<double[][]> normalizedByClass;
            ArrayList<double[][]> originalByClass;

            if (DV.classification)
            {
                // separate by class
                normalizedByClass = separateByClass(normalizedNumericalData, allClasses);
                originalByClass = separateByClass(originalNumericalData, allClasses);
            }
            else
            {
                normalizedByClass = new ArrayList<>();
                normalizedByClass.add(normalizedNumericalData);
                originalByClass = new ArrayList<>();
                originalByClass.add(originalNumericalData);
            }

            // transform classes into data objects
            DV.trainData = createDataObjects(normalizedByClass);
            DV.originalData = createDataObjects(originalByClass);
        }
        else
        {
            DV.trainData = createDataObjects(new ArrayList<>(Arrays.asList(normalizedNumericalData, new double[0][0])));
            DV.originalData = createDataObjects(new ArrayList<>(Arrays.asList(originalNumericalData, new double[0][0])));
        }

        // points to be highlighted
        DV.highlights = new boolean[DV.classNumber][];
        for (int i = 0; i < DV.classNumber; i++)
        {
            DV.highlights[i] = new boolean[DV.trainData.get(i).data.length];
            for (int j = 0; j < DV.trainData.get(i).data.length; j++)
                DV.highlights[i][j] = false;
        }

        createTrainTestSplit();

        return true;
    }


    /**
     * Sets up validation data for user made worst case confusion matrix
     * @param valFile .csv file holding data
     * @return whether setupValidationData was successful
     */
    public static boolean setupValidationData(File valFile)
    {
        if (DV.hasClasses)
        {
            // data string[][] representation of import file
            String[][] stringData = getStringFromCSV(valFile);
            ArrayList<String> validationClasses;

            // check for proper format
            if (checkFormat(stringData))
            {
                if (stringData != null)
                {
                    // check classes and update class number
                    validationClasses = checkAllClasses(stringData);
                    if (!validationClasses.isEmpty())
                        stringData = purgeClasses(stringData);
                    else
                        return false;

                    // remove ID
                    if (DV.hasID)
                        stringData = purgeID(stringData);

                    // get numerical data from string data
                    double[][] numericalData = stringToNumerical(stringData);
                    if (numericalData != null)
                    {
                        // normalize data
                        // transform classes into data objects
                        double[][] normalizedNumericalData = normalizeSecondaryData(numericalData);
                        ArrayList<double[][]> splitByClass = separateByClass(normalizedNumericalData, validationClasses);
                        DV.testData = createDataObjects(splitByClass);

                        return true;
                    }
                    else
                        return false;
                }
                else
                    return false;
            }
            else
                return false;
        }
        else
            return false;
    }


    /**
     * Checks if the format of the input data
     * matches that of previous data.
     * @param stringData String[][] representation of input data
     * @return whether the format was valid or not
     */
    private static boolean checkFormat(String[][] stringData)
    {
        // get number of columns for data
        int cols = DV.fieldLength;
        if (DV.hasID) cols++;
        if (DV.hasClasses) cols++;

        return stringData[0].length == cols;
    }


    /**
     * Sets up DV with imported data from importFile
     * @param importFile .csv file holding data
     * @return whether setupImportData was successful
     */
    public static boolean setupImportData(File importFile)
    {
        // data string[][] representation of import file
        String[][] stringData = getStringFromCSV(importFile);

        // check for proper format
        if (checkFormat(stringData))
        {
            if (stringData != null)
            {
                for (int i = 0; i < DV.dataFiles.size(); i++)
                {
                    String[][] otherData = getStringFromCSV(DV.dataFiles.get(i));
                    if (otherData != null)
                        stringData = combine2DArrays(stringData, otherData);
                    else
                        return false;
                }

                return setupWithData(stringData);
            }
            else
                return false;
        }
        else
            return false;
    }


    /**
     * Set up SVM Support Vectors
     * @param svFile file containing support vectors
     * @return whether setupSupportVectors was successful
     */
    public static boolean setupSupportVectors(File svFile)
    {
        // data string[][] representation of dataFile (csv)
        String[][] stringData = getStringFromCSV(svFile);
        if (stringData != null)
        {
            // get numerical data from string data
            double[][] numericalData = stringToNumerical(stringData);

            // get normalized numerical data if not null
            if (numericalData != null)
            {
                // normalize data
                double[][] normalizedNumericalData = normalizeData(numericalData);
                DV.supportVectors = createDataObjects(new ArrayList<>(Arrays.asList(normalizedNumericalData, new double[0][0]))).get(0);

                return true;
            }
            else
                return false;
        }
        else
            return false;
    }


    /**
     * Sets up DV with saved project
     * @param projectFile .csv file holding project info
     */
    public static void setupProjectData(File projectFile)
    {
        // data string[][] representation of project file
        String[][] stringData = getStringFromCSV(projectFile);

        if (stringData != null)
        {
            int buffer = 0;

            // get graph colors
            DV.graphColors[0] = new Color(Integer.parseInt(stringData[buffer][0]), Integer.parseInt(stringData[buffer][1]), Integer.parseInt(stringData[buffer++][2]));
            DV.graphColors[1] = new Color(Integer.parseInt(stringData[buffer][0]), Integer.parseInt(stringData[buffer][1]), Integer.parseInt(stringData[buffer++][2]));
            DV.background = new Color(Integer.parseInt(stringData[buffer][0]), Integer.parseInt(stringData[buffer][1]), Integer.parseInt(stringData[buffer++][2]));

            // get line colors
            DV.domainLines = new Color(Integer.parseInt(stringData[buffer][0]), Integer.parseInt(stringData[buffer][1]), Integer.parseInt(stringData[buffer++][2]));
            DV.overlapLines = new Color(Integer.parseInt(stringData[buffer][0]), Integer.parseInt(stringData[buffer][1]), Integer.parseInt(stringData[buffer++][2]));
            DV.thresholdLine = new Color(Integer.parseInt(stringData[buffer][0]), Integer.parseInt(stringData[buffer][1]), Integer.parseInt(stringData[buffer++][2]));

            // get data format
            DV.hasID = "1".equals(stringData[buffer][0]);
            DV.hasClasses = "1".equals(stringData[buffer][1]);
            DV.zScoreMinMax = "1".equals(stringData[buffer++][2]);

            // get field length
            DV.fieldLength = Integer.parseInt(stringData[buffer++][0]);

            // get angles
            DV.angles = new double[DV.fieldLength];

            for (int i = 0; i < DV.fieldLength; i++)
                DV.angles[i] = Double.parseDouble(stringData[buffer][i]);

            buffer++;

            // get threshold
            DV.threshold = Double.parseDouble(stringData[buffer++][0]);
            DV.thresholdSlider.setValue((int) (DV.threshold / DV.fieldLength * 200) + 200);

            // get overlap area
            DV.overlapArea = new double[]{ Double.parseDouble(stringData[buffer][0]), Double.parseDouble(stringData[buffer++][1]) };
            DV.overlapSlider.setValue((int) (DV.overlapArea[0] / DV.fieldLength * 200) + 200);
            DV.overlapSlider.setUpperValue((int) (DV.overlapArea[1] / DV.fieldLength * 200) + 200);

            // get domain area
            DV.domainArea = new double[]{ Double.parseDouble(stringData[buffer][0]), Double.parseDouble(stringData[buffer++][1]) };
            DV.domainSlider.setValue((int) (DV.domainArea[0] / DV.fieldLength * 200) + 200);
            DV.domainSlider.setUpperValue((int) (DV.domainArea[1] / DV.fieldLength * 200) + 200);

            // save analytics toggles
            DV.prevAllDataChecked = "1".equals(stringData[buffer][0]);
            DV.allDataChecked = "1".equals(stringData[buffer][1]);
            DV.withoutOverlapChecked = "1".equals(stringData[buffer][2]);
            DV.overlapChecked = "1".equals(stringData[buffer][3]);
            DV.worstCaseChecked = "1".equals(stringData[buffer][4]);
            DV.userValidationChecked = "1".equals(stringData[buffer][5]);
            DV.userValidationImported = "1".equals(stringData[buffer][6]);
            DV.crossValidationChecked = "1".equals(stringData[buffer++][7]);

            // get previous confusion matrices
            int prevCMs = Integer.parseInt(stringData[buffer++][0]);

            for (int i = 0; i < prevCMs; i++)
            {
                char[] cm = stringData[buffer++][0].toCharArray();

             for (int j = 0; j < cm.length; j++)
             {
                 // replace placeholder character with newline
                 if (cm[j] == '~')
                     cm[j] = '\n';
                 else if (cm[j] == '`')
                     cm[j] = ',';
             }

                DV.prevAllDataCM.add(new String(cm));
            }

            // get k-folds
            DV.kFolds = Integer.parseInt(stringData[buffer++][0]);

            // get number of classes
            DV.classNumber = Integer.parseInt(stringData[buffer++][0]);

            // get visualized classes
            DV.upperClass = Integer.parseInt(stringData[buffer++][0]);

            DV.lowerClasses = new ArrayList<>();

            for (int i = 0; i < DV.classNumber; i++)
            {
                if ("1".equals(stringData[buffer][i]))
                    DV.lowerClasses.add(true);
                else
                    DV.lowerClasses.add(false);
            }

            buffer++;

            // get class order
            DV.upperIsLower = "1".equals(stringData[buffer++][0]);

            // get unique classes
            DV.uniqueClasses = new ArrayList<>(Arrays.asList(stringData[buffer++]).subList(0, DV.classNumber));

            // get fieldNames
            DV.fieldNames = new ArrayList<>(Arrays.asList(stringData[buffer++]).subList(0, DV.fieldLength));

            // get data
            ArrayList<double[][]> classData = new ArrayList<>();

            for (int c = 0; c < DV.classNumber; c++)
            {
                // get number of data points
                int numPoints = Integer.parseInt(stringData[buffer++][0]);

                // array to hold class data
                double[][] normData = new double[numPoints][DV.fieldLength];

                for (int i = 0; i < numPoints; i++)
                {
                    for (int j = 0; j < DV.fieldLength; j++)
                        normData[i][j] = Double.parseDouble(stringData[buffer][j]);

                    buffer++;
                }

                classData.add(normData);
            }

            // create DataObjects
            DV.trainData = createDataObjects(classData);
            classData.clear();

            // get original data
            for (int c = 0; c < DV.classNumber; c++)
            {
                // get number of data points
                int numPoints = Integer.parseInt(stringData[buffer++][0]);

                // array to hold class data
                double[][] origData = new double[numPoints][DV.fieldLength];

                for (int i = 0; i < numPoints; i++)
                {
                    for (int j = 0; j < DV.fieldLength; j++)
                        origData[i][j] = Double.parseDouble(stringData[buffer][j]);

                    buffer++;
                }

                classData.add(origData);
            }

            // create DataObjects
            DV.originalData = createDataObjects(classData);
            classData.clear();

            if (DV.userValidationImported)
            {
                // get validation data data
                for (int c = 0; c < DV.classNumber; c++)
                {
                    // get number of data points
                    int numPoints = Integer.parseInt(stringData[buffer++][0]);

                    // array to hold class data
                    double[][] valData = new double[numPoints][DV.fieldLength];

                    for (int i = 0; i < numPoints; i++)
                    {
                        for (int j = 0; j < DV.fieldLength; j++)
                            valData[i][j] = Double.parseDouble(stringData[buffer][j]);

                        buffer++;
                    }

                    classData.add(valData);
                }

                // create DataObjects
                DV.validationData = createDataObjects(classData);
            }
        }
    }


    /**
     * Gets all classes from input data
     * @param stringData String[][] representation of input data
     * @return found classes
     */
    private static ArrayList<String> getClasses(String[][] stringData)
    {
        // LinkedHashSet preserves order and does not allow duplicates
        // ArrayList to store the class of each datapoint
        LinkedHashSet<String> unique;
        ArrayList<String> classes = new ArrayList<>();

        // add already known classes if importing data
        if (DV.uniqueClasses != null)
            unique = new LinkedHashSet<>(DV.uniqueClasses);
        else
            unique = new LinkedHashSet<>();

        // add all classes to unique and allClasses
        for (int i = 1; i < stringData.length; i++)
        {
            unique.add(stringData[i][stringData[0].length - 1]);
            classes.add(stringData[i][stringData[0].length - 1]);
        }

        // store unique classes and class number
        DV.uniqueClasses = new ArrayList<>(unique);
        DV.classNumber = unique.size();

        return classes;
    }


    /**
     * Checks if validation classes are valid. (same classes as visualized data)
     * Stores classes if valid.
     * @param stringData String[][] representation of input data
     * @return whether classes were valid
     */
    private static ArrayList<String> checkAllClasses(String[][] stringData)
    {
        // check all classes
        boolean valid = true;
        for (int i = 1; i < stringData.length; i++)
        {
            if (valid)
            {
                for (int j = 0; j < DV.uniqueClasses.size(); j++)
                {
                    // ensure classes are the same
                    if (stringData[i][stringData[i].length - 1].equals(DV.uniqueClasses.get(j)))
                        break;
                    else if (j == DV.uniqueClasses.size() - 1)
                        valid = false;
                }
            }
        }

        // store validation classes
        // add all classes to unique and allClasses
        ArrayList<String> validationClasses = new ArrayList<>();
        if (valid)
        {
            for (int i = 1; i < stringData.length; i++)
                validationClasses.add(stringData[i][stringData[0].length - 1]);
        }
            return validationClasses;
    }


    /**
     * Search Terms: Reads from csv. Reads in data. Reads data from csv. Read from CSV. read csv
     * @param dataFile .csv file holding data
     * @return String[][] representation of dataFile
     */
    public static String[][] getStringFromCSV(File dataFile) {
        //long startTime = System.nanoTime();

        // Executor service to handle parallel tasks
        ExecutorService executorService = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

        try (Scanner fileReader = new Scanner(dataFile)) {
            List<Callable<String[]>> tasks = new ArrayList<>();
            List<String> rowData = new ArrayList<>();

            // Read all rows and store them
            while (fileReader.hasNextLine()) {
                rowData.add(fileReader.nextLine());
            }

            // Split the rows in parallel
            for (String row : rowData) {
                tasks.add(() -> row.split(","));
            }

            // Submit tasks for parallel execution
            List<Future<String[]>> results = executorService.invokeAll(tasks);

            // Collect the results
            List<String[]> data = new ArrayList<>();
            for (Future<String[]> result : results) {
                try {
                    String[] row = result.get();  // Get the result of the task
                    if (row.length > 0) {
                        data.add(row);
                    }
                } catch (InterruptedException | ExecutionException e) {
                    e.printStackTrace();
                }
            }

            // Convert list of arrays to 2D array
            String[][] outputData = new String[data.size()][];
            data.toArray(outputData);
            //long endTime = System.nanoTime();
            //long elapsedTime = (endTime - startTime) / 1_000_000; // Convert to milliseconds
            //System.out.println("Execution time for parallel reading: " + elapsedTime + " milliseconds");
            return outputData;

        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
            return null;
        } finally {
            // Shutdown the executor service
            executorService.shutdown();
        }
    }


    /**
     * Removes ID column from stringData
     * @param stringData String[][] representation of input data
     * @return String[][] representation of input data without ID column
     */
    private static String[][] purgeID(String[][] stringData)
    {
        // copies every value except the last column
        String[][] noID = new String[stringData.length][stringData[0].length - 1];
        for (int i = 0; i < stringData.length; i++)
        {
            if (stringData[0].length - 1 >= 0)
                System.arraycopy(stringData[i], 1, noID[i], 0, stringData[i].length - 1);
        }

        return noID;
    }


    /**
     * Removes class column from stringData
     * @param stringData String[][] representation of input data
     * @return String[][] representation of input data without class column
     */
    private static String[][] purgeClasses(String[][] stringData)
    {
        // copies every value except the last column
        String[][] noClasses = new String[stringData.length][stringData[0].length - 1];
        for (int i = 0; i < stringData.length; i++)
        {
            if (stringData[0].length - 1 >= 0)
                System.arraycopy(stringData[i], 0, noClasses[i], 0, stringData[i].length - 1);
        }

        // normalize regression target values
        if (!DV.classification)
        {
            double[][] numericalData = new double[stringData.length-1][1];

            for (int i = 0; i < stringData.length-1; i++)
                numericalData[i][0] = Double.parseDouble(stringData[i+1][stringData[i+1].length-1]);

            double[][] normalizedData = normalizeData(numericalData);
            for (double[] normalizedDatum : normalizedData)
                DV.reg_true_values.add(normalizedDatum[0]);
        }

        return noClasses;
    }


    /**
     * Stores first row of dataset as field names
     * @param stringData String[][] representation of input data
     * @return ArrayList with field names
     */
    private static ArrayList<String> getFieldNames(String[][] stringData)
    {
        return new ArrayList<>(List.of(stringData[0]));
    }


    /**
     * Transforms string data into numerical data
     * @param stringData String[][] representation of input data
     * @return normalized input data
     */
    private static double[][] stringToNumerical(String[][] stringData)
    {
        // ArrayList to store numerical data
        ArrayList<ArrayList<Double>> numericalData = new ArrayList<>();
        for (int i = 0; i < stringData.length - 1; i++)
            numericalData.add(new ArrayList<>());

        // true if asked to keep invalid data
        // number of invalid data
        boolean askedToKeep = false;
        int invalids = 0;

        // first row is for field names: skip
        // convert every value in every row to double
        for (int i = 0; i < stringData.length - 1; i++)
        {
            for (int j = 0; j < stringData[i].length; j++)
            {
                // if value cannot become double then quit or remove row
                try
                {
                    numericalData.get(i-invalids).add(Double.parseDouble(stringData[i+1][j]));
                }
                catch (NumberFormatException nfe)
                {
                    if (!askedToKeep)
                    {
                        // inform user about invalid data
                        // return if canceled
                        if(stringToNumericalHelper1(i, j) == 1)
                            return null;

                        // don't ask again
                        askedToKeep = true;
                    }

                    if (DV.hasClasses)
                        stringToNumericalHelper2(i, invalids);

                    // remove invalid ArrayList from numerical data
                    numericalData.remove(i - invalids++);
                    break;
                }
            }
        }

        if (!numericalData.isEmpty())
        {
            // convert numericalData to double[][]
            double[][] outputData = new double[numericalData.size()][];

            for (int i = 0; i < numericalData.size(); i++)
                outputData[i] = numericalData.get(i).stream().mapToDouble(m -> m).toArray();

            return outputData;
        }
        else
            return null;
    }


    /**
     * Helper function for stringToNumerical
     * @param i row
     * @param j column
     * @return user choice
     */
    private static int stringToNumericalHelper1(int i, int j)
    {
        String message;
        if(DV.hasID)
        {
            message = String.format("Invalid data found in column %d row %d.\n" +
                    "Would you like to remove all rows with invalid data or cancel visualization?", j+2, i+2);
        }
        else
        {
            message = String.format("Invalid data found in column %d row %d.\n" +
                    "Would you like to remove all rows with invalid data or cancel visualization?", j+1, i+2);
        }

        Object[] buttons = {"Remove invalids", "Cancel Visualization"};

        return JOptionPane.showOptionDialog(DV.mainFrame,
                message,
                "Invalid Data Found",
                JOptionPane.YES_NO_OPTION,
                JOptionPane.ERROR_MESSAGE,
                null,
                buttons,
                null);
    }


    /**
     * Helper function for stringToNumerical
     * @param i row
     * @param invalids number of invalids
     */
    private static void stringToNumericalHelper2(int i, int invalids)
    {
        // remove invalid from allClasses
        String invalidData = allClasses.get(i - invalids);
        allClasses.remove(i - invalids);

        // check if class still exists
        for (int k = 0; k < allClasses.size(); k++)
        {
            if (invalidData.equals(allClasses.get(k)))
                break;
            else if (k == allClasses.size() - 1)
            {
                DV.uniqueClasses.remove(invalidData);
                DV.classNumber--;
            }
        }
    }


    /**
     * Normalized input data with Min-Max
     * or z-Score Min-Max normalization
     * @param data input data
     * @return normalized input data
     */
    private static double[][] normalizeData(double[][] data)
    {
        // min and max values of each column
        maxValues = new double[data[0].length];
        minValues = new double[data[0].length];

        // do z-Score Min-Max or Min-Max normalization
        if (DV.zScoreMinMax)
            zScore(data);

        // get min and max values
        for (int i = 0; i < data[0].length; i++)
        {
            // set max and min
            maxValues[i] = data[0][i];
            minValues[i] = data[0][i];
            for (int j = 1; j < data.length; j++)
            {
                // check for better max or min
                if (data[j][i] > maxValues[i])
                    maxValues[i] = data[j][i];
                else if (data[j][i] < minValues[i])
                    minValues[i] = data[j][i];
            }
        }

        for (int i = 0; i < data[0].length; i++)
        {
            if (maxValues[i] != minValues[i])
            {
                for (int j = 0; j < data.length; j++)
                    data[j][i] = (data[j][i] - minValues[i]) / (maxValues[i] - minValues[i]);
            }
            else
            {
                if (!setValues)
                {
                    String message = String.format("Minimum and maximum values in column %d are the same.\n" +
                            "Would you like to manually enter a minimum and maximum or default all values to 0.5?", i+1);

                    // ask user for manual min max entry
                    double[] manualMinMax = manualMinMaxEntry(message);

                    // use manual min max or default to 0.5 if null
                    if (manualMinMax != null)
                    {
                        for (int j = 0; j < data.length; j++)
                            data[j][i] = (data[j][i] - manualMinMax[0]) / (manualMinMax[1] - manualMinMax[0]);
                    }
                    else
                    {
                        for (int j = 0; j < data.length; j++)
                            data[j][i] = 0.5;
                    }
                }
                else
                {
                    for (int j = 0; j < data.length; j++)
                        data[j][i] = 0.5;
                }
            }
        }

        return data;
    }


    /**
     * Perform z-Score normalization on input data
     * @param data input data
     */
    private static void zScore(double[][] data)
    {
        // mean and standard deviation per column
        mean = new double[data[0].length];
        sd = new double[data[0].length];

        // get mean for each column
        for (int i = 0; i < data[0].length; i++)
        {
            for (double[] dataPoint : data)
                mean[i] += dataPoint[i];

            mean[i] /= data.length;
        }

        // get standard deviation for each column
        for (int i = 0; i < data[0].length; i++)
        {
            for (double[] dataPoint : data)
                sd[i] += Math.pow(dataPoint[i] - mean[i], 2);

            sd[i] = Math.sqrt(sd[i] / data.length);

            if (sd[i] < 0.001)
            {
                String message = String.format("""
                            Standard deviation in column %d is less than 0.001.
                            Please manually enter a minimum and maximum. The standard deviation will get(max - min) / 2.
                            Else, the standard deviation will get 0.001.""", i+1);

                // ask user for manual min max entry
                double[] manualMinMax = manualMinMaxEntry(message);

                // use manual min max or default to 0.001 if null
                if (manualMinMax != null)
                    sd[i] = (manualMinMax[1] / manualMinMax[0]) / 2;
                else
                    sd[i] = 0.001;
            }
        }

        // get min and max values
        for (int i = 0; i < data[0].length; i++)
        {
            // perform z-Score
            for (int j = 0; j < data.length; j++)
                data[j][i] = (data[j][i] - mean[i]) / sd[i];
        }
    }


    /**
     * Normalized input data with Min-Max
     * or z-Score Min-Max normalization
     * Uses z-Score and Min-Max values from the normalizeData Function
     * @param data input data
     * @return normalized input data
     */
    private static double[][] normalizeSecondaryData(double[][] data)
    {
        // normalize all data
        // get data lengths
        int newDataLength = data.length;
        int allDataLength = newDataLength;
        for (int i = 0; i < DV.trainData.size(); i++)
            allDataLength += DV.trainData.get(i).data.length;

        // get all data
        int cnt = 0;
        double[][] allData = new double[allDataLength][DV.fieldLength];
        for (int i = 0; i < DV.originalData.size(); i++)
        {
            System.arraycopy(DV.originalData.get(i).data, 0, allData, cnt, DV.originalData.get(i).data.length);
            cnt += DV.originalData.get(i).data.length;
        }

        System.arraycopy(data, 0, allData, cnt, newDataLength);

        // normalize all data
        double[][] allNormalizedData = normalizeData(allData);

        // separate new data
        double[][] newData = new double[newDataLength][DV.fieldLength];
        for (int i = 0; i < newDataLength; i++)
        {
            System.arraycopy(allNormalizedData[cnt], 0, newData[i], 0, DV.fieldLength);
            cnt++;
        }

        // separate old data
        cnt = 0;
        for (int i = 0; i < DV.trainData.size(); i++)
        {
            double[][] oldData = new double[DV.trainData.get(i).data.length][DV.fieldLength];
            for (int j = 0; j < DV.trainData.get(i).data.length; j++)
                System.arraycopy(allNormalizedData[j + cnt], 0, oldData[j], 0, DV.fieldLength);

            DV.trainData.set(i, new DataObject(DV.trainData.get(i).className, oldData));
            cnt += DV.trainData.get(i).data.length;
        }

        return newData;
    }


    /**
     * Separates data by class
     * @param data data to be separated
     * @param classes classes of every datapoint
     * @return ArrayList of separate class data
     */
    private static ArrayList<double[][]> separateByClass(double[][] data, ArrayList<String> classes)
    {
        // output ArrayList
        // holds arraylists holding data-points for each class
        ArrayList<double[][]> separatedClasses = new ArrayList<>();
        ArrayList<ArrayList<double[]>> divider = new ArrayList<>();

        for (int i = 0; i < DV.classNumber; i++)
            divider.add(new ArrayList<>());

        // check every class
        for (int i = 0; i < classes.size(); i ++)
        {
            // find matching class
            for (String str : DV.uniqueClasses)
            {
                if (str.equals(classes.get(i)))
                {
                    divider.get(DV.uniqueClasses.indexOf(str)).add(data[i]);
                    break;
                }
            }
        }

        // transform to double[][]
        for (ArrayList<double[]> separateClass : divider)
            separatedClasses.add(separateClass.toArray(new double[separateClass.size()][]));

        return separatedClasses;
    }


    /**
     * Transforms ArrayList<double[][]> into DataObjects
     * @param separateClasses ArrayList to transform into DataObjects
     * @return ArrayList of DataObjects
     */
    public static ArrayList<DataObject> createDataObjects(ArrayList<double[][]> separateClasses)
    {
        // get classes and output arraylist
        ArrayList<DataObject> dataObjects = new ArrayList<>();

        // creates DataObject with class name and data
        for (int i = 0; i < separateClasses.size(); i++)
        {
            if (separateClasses.get(i).length > 0)
                dataObjects.add(new DataObject(DV.uniqueClasses.get(i), separateClasses.get(i)));
        }

        return dataObjects;
    }


    /**
     * Gets user entry for min and max values
     * @param message Message to display to users
     * @return min and max values
     */
    public static double[] manualMinMaxEntry(String message)
    {
        // ask user for manual entry or default column to 0.5
        Object[] options = { "Manual Entry", "Default Column to 0.5", "Default all similar Columns to 0.5" };
        int choice = JOptionPane.showOptionDialog(
                DV.mainFrame,
                message,
                "Numerical Error",
                JOptionPane.YES_NO_OPTION,
                JOptionPane.ERROR_MESSAGE,
                null,
                options,
                null);

        // choice == manual entry
        if (choice == 0)
            return manualMinMaxEntryHelper1();
        else if (choice == 2)
            setValues = true;

        return null;
    }


    /**
     * Helper function for manualMinMaxEntry
     * @return min and max values
     */
    private static double[] manualMinMaxEntryHelper1()
    {
        // popup asking for min and max input
        JPanel minMaxPanel = new JPanel(new BorderLayout());
        minMaxPanel.add(new JLabel("Enter a Minimum and Maximum value or cancel to default to 0.5."), BorderLayout.NORTH);

        // text panel
        JPanel textPanel = new JPanel(new FlowLayout(FlowLayout.CENTER));

        // maximum text field
        JTextField maxField = new JTextField();
        maxField.setPreferredSize(new Dimension(30, 30));
        textPanel.add(new JLabel("Maximum"));
        textPanel.add(maxField);

        // minimum text field
        JTextField minField = new JTextField();
        minField.setPreferredSize(new Dimension(30, 30));
        textPanel.add(new JLabel("Minimum"));
        textPanel.add(minField);

        // add text panel
        minMaxPanel.add(textPanel, BorderLayout.SOUTH);

        // loop until min and max are valid or user quits
        while (true)
        {
            int choice = JOptionPane.showConfirmDialog(DV.mainFrame, minMaxPanel, "Enter Minimum and Maximum", JOptionPane.OK_CANCEL_OPTION, JOptionPane.PLAIN_MESSAGE);
            if (choice == 0)
            {
                double[] minMax = manualMinMaxEntryHelper2(minField.getText(), maxField.getText());
                if (minMax.length > 0)
                    return minMax;
            }
            else
                return null;
        }
    }


    /**
     * Helper function for manualMinMaxEntry
     * @param minField inputted minimum value
     * @param maxField inputted maximum value
     * @return min and max values
     */
    private static double[] manualMinMaxEntryHelper2(String minField, String maxField)
    {
        try
        {
            // get text field values
            double max = Double.parseDouble(maxField);
            double min = Double.parseDouble(minField);

            if (min < max)
            {
                // return min and max
                return new double[] { min, max };
            }
            else
            {
                // min is greater than or equal to max
                DV.warningPopup(
                        "Error",
                        "Error: minimum is greater than or equal to maximum.\n" +
                                "Please ensure the minimum is less than the maximum.");
            }
        }
        catch (NumberFormatException nfe)
        {
            DV.warningPopup("Error", "Error: please enter numerical values.");
        }

        return new double[0];
    }


    /**
     * Gets train test split from data
     */
    private static void createTrainTestSplit()
    {
        // instantiate testing data
        DV.testData = new ArrayList<>();
        for (int i = 0; i < DV.classNumber; i++)
        {
            // randomize order
            DV.trainData.get(i).data = fisher_yates_shuffle(DV.trainData.get(i).data);

            // get size of training data
            int trainSize = (int) Math.floor(DV.trainData.get(i).data.length * DV.trainSplit);
            double[][] training = new double[trainSize][DV.fieldLength];

            // get training data
            for (int j = 0; j < trainSize; j++)
                System.arraycopy(DV.trainData.get(i).data[j], 0, training[j], 0, DV.fieldLength);

            // get size of testing data
            int testSize = DV.trainData.get(i).data.length - trainSize;
            double[][] testing = new double[testSize][DV.fieldLength];

            // get testing data
            for (int j = trainSize; j < DV.trainData.get(i).data.length; j++)
                System.arraycopy(DV.trainData.get(i).data[j], 0, testing[j - trainSize], 0, DV.fieldLength);

            // store training and testing data
            DV.trainData.set(i, new DataObject(DV.uniqueClasses.get(i), training));
            DV.testData.add(new DataObject(DV.uniqueClasses.get(i), testing));
        }
    }


    /**
     * Fisher-Yates Shuffle
     * @param data array to be shuffled
     * @return shuffled data
     */
    private static double[][] fisher_yates_shuffle(double[][] data)
    {
        Random rand = new Random();
        for (int i = data.length - 1; i > 0; i--)
        {
            // generate random index
            int j = rand.nextInt(i+1);

            // swap
            double[] tmp = data[i].clone();
            data[i] = data[j];
            data[j] = tmp;
        }

        // return shuffled array
        return data;
    }


    /**
     * Combines two 2D arrays
     * @param array1 first array
     * @param array2 second array
     * @return combined array
     */
    public static String[][] combine2DArrays(String[][] array1, String[][] array2)
    {
        int totalRows = array1.length + array2.length;
        int columns = array1[0].length; // Assuming both arrays have the same number of columns

        String[][] combinedArray = new String[totalRows][columns];

        // Copy elements from the first array
        for (int i = 0; i < array1.length; i++)
            System.arraycopy(array1[i], 0, combinedArray[i], 0, columns);

        // Copy elements from the second array
        for (int i = 0; i < array2.length; i++)
            System.arraycopy(array2[i], 0, combinedArray[array1.length + i], 0, columns);

        return combinedArray;
    }
}
