import java.util.ArrayList;
import java.util.List;

public class HyperBlock
{
    /*
        Supposed to hold all hyper-blocks data seperated by the outer list. Instead, holds only 1.
        ArrayList<double[]> should hold all datapoints in the block. ex {[x1,x2,x3], [x1,x2,x3]}
     */
    ArrayList<ArrayList<double[]>> hyper_block;

    // class number of hyperblock
    int classNum;

    // number of datapoints in hyperblock
    int size;

    // class name of hyperblock
    String className;

    // seed attribute of hyperblocks created with IHyper algorithm
    String attribute;

    // minimums and maximums for each feature in hyperblock
    ArrayList<ArrayList<Double>> maximums;
    ArrayList<ArrayList<Double>> minimums;

    /**
     * Constructor for HyperBlock
     * @param hyper_block datapoints to go in hyperblock
     */
    HyperBlock(ArrayList<ArrayList<double[]>> hyper_block, int classNum)
    {
        this.hyper_block = hyper_block;

        this.maximums = new ArrayList<>();
        this.minimums = new ArrayList<>();
        this.classNum = classNum;

        findBounds();
    }

    /**
     * This will return the number of intervals that the HyperBlock has on the attribute specified.
     * @param attributeNumber The index of the attribute (int).
     */
    public int intervalCount(int attributeNumber){
        int n = this.maximums.get(attributeNumber).size();

        if(n == this.minimums.get(attributeNumber).size()){
            return n;
        }

        return -1;
    }

    /**
     * Return the max number of intervals that the block has on any attribute.
     * Ex. if it has 1 interval for x1, but 3 for x2 it will return 3.
     * @return
     */
    public int getMaxDisjunctiveORs(){
        int max = Integer.MIN_VALUE;

        for(int i = 0; i < maximums.size(); i++){
            max = Math.max(max, maximums.get(i).size());
        }

        return max;
    }


    /**
     * Constructor for HyperBlock
     * @param hyper_block datapoints to go in hyperblock
     */
    HyperBlock(ArrayList<ArrayList<double[]>> hyper_block)
    {
        this.hyper_block = hyper_block;

        this.maximums = new ArrayList<>();
        this.minimums = new ArrayList<>();

        findBounds();
    }


    /**
     * Constructor for HyperBlock
     * @param max maximum bound for hyperblock
     * @param min minimum bound for hyperblock
     */
    HyperBlock(ArrayList<ArrayList<Double>> max, ArrayList<ArrayList<Double>> min, int classNum)
    {
        setBounds(max, min);
        findData();
        this.classNum = classNum;
    }


    /**
     * Constructor for HyperBlock
     * @param max maximum bound for hyperblock
     * @param min minimum bound for hyperblock
     */
    HyperBlock(ArrayList<ArrayList<Double>> max, ArrayList<ArrayList<Double>> min)
    {
        setBounds(max, min);
        findData();
    }

    /**
     * Sets minimum and maximum bound for a hyperblock
     * @param max maximum bound
     * @param min minimum bound
     */
    private void setBounds(ArrayList<ArrayList<Double>> max, ArrayList<ArrayList<Double>> min)
    {
        // Make the outer list for max then add the inner list elements.
        maximums = new ArrayList<>();
        for (ArrayList<Double> innerList : max) {
            maximums.add(new ArrayList<>(innerList));
        }

        minimums = new ArrayList<>();
        for (ArrayList<Double> innerList : min) {
            minimums.add(new ArrayList<>(innerList));
        }
    }

    /**
     * Finds all data within a hyperblock
     */
    private void findData()
    {
        // The datapoints
        ArrayList<ArrayList<double[]>> dps = new ArrayList<>();
        ArrayList<double[]> classPnts = new ArrayList<>();

        // Go through each class in the dataset.
        for (int i = 0; i < DV.trainData.size(); i++)
        {
            // Go through data point in the class of the dataset
            for (double[] point : DV.trainData.get(i).data)
            {
                boolean inside = true;
                //System.out.println(Arrays.toString(point));
                // Go through all the attributes mins/maxes
                for (int k = 0; k < DV.fieldLength; k++)
                {
                    boolean inAnyInterval = false;
                    double value = point[k];
                    // Check all intervals for the current attribute
                    for (int g = 0; g < maximums.get(k).size(); g++) {

                        // Check if the value is within the interval
                        if (value >= minimums.get(k).get(g) && value <= maximums.get(k).get(g)) {
                            inAnyInterval = true;
                            break;
                        }
                    }

                    // If it's not in any interval for one of the attributes it isnt in the block
                    if(!inAnyInterval){
                        inside = false;
                        break;
                    }
                }

                if (inside){
                    classPnts.add(point);
                }
            }
        }

        dps.add(classPnts);
        hyper_block = dps;
    }

    /**
     * Gets minimums and maximums of a hyper-block
     *
     * Assumes non-disjunctive blocks for now.
     */
    public void findBounds()
    {
        size = 0;
        maximums.clear();
        minimums.clear();

        // Initialize the maximums and minimums with one interval for each attribute
        for (int g = 0; g < DV.fieldLength; g++) {
            maximums.add(new ArrayList<>(List.of(Double.NEGATIVE_INFINITY)));
            minimums.add(new ArrayList<>(List.of(Double.POSITIVE_INFINITY)));
        }

        // Go through all the points in the hyperblock
        for (double[] dbs : hyper_block.get(0))
        {
            for (int j = 0; j < DV.fieldLength; j++)
            {
                // Update dimensional min or max if the value is < or >
                maximums.get(j).set(0, Math.max(maximums.get(j).get(0), dbs[j]));
                minimums.get(j).set(0, Math.min(minimums.get(j).get(0), dbs[j]));

            }
        }
        // From old version, don't want to try removing rn
        size += hyper_block.get(0).size();
    }
}
