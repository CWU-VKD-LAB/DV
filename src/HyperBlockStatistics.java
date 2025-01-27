import javax.swing.*;
import javax.swing.border.EmptyBorder;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.lang.reflect.Array;
import java.util.*;
import java.util.List;
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


    final int SIMP_PANELS_FONTSIZE = 14;
    private JPanel simpPanel;
    private JPanel clausePanel;
    private JPanel attrPanel;
    private JPanel blockPanel;
    private JFrame mainFrame;
    private JFrame attributeVisWindow;

    private int firstSetIdx = -1;
    private int secondSetIdx =-1;
    private boolean showPercent = true;

    private ArrayList<HyperBlock> hyper_blocks;
    private ArrayList<DataObject> data;
    private HyperBlockGeneration hbGen;

    private final JLabel[] beforeLabels = new JLabel[4 + DV.uniqueClasses.size()];


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

    boolean debug = false;
    boolean console = false;

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
        mainFrame  = new JFrame();
        mainFrame.setDefaultCloseOperation(JFrame.HIDE_ON_CLOSE);
        mainFrame.setLayout(new BorderLayout());


        // Dataset info footer.
        int len = 0;
        for(int i = 0; i < data.size();i++){
            len += data.get(i).data.length;
        }
        String label = "Dataset: " + DV.dataFileName + "   Points: " + len + "   Dimensions: " + DV.fieldLength + "-D";
        JLabel dataSetInfoLbl = new JLabel(label, SwingConstants.CENTER);
        dataSetInfoLbl.setFont(new Font("Arial", Font.BOLD, 14));
        dataSetInfoLbl.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));
        mainFrame.add(dataSetInfoLbl, BorderLayout.SOUTH);


        mainFrame.add(createToolBar(), BorderLayout.NORTH);



        //Make all the panels
        simpPanel = createSimplificationPanel();

        simpPanel.setPreferredSize(new Dimension(800, 400));
        mainFrame.add(simpPanel, BorderLayout.CENTER);


        // Show the frame
        //mainFrame.setSize(400, 200);
        mainFrame.setVisible(true);
        mainFrame.revalidate();
        mainFrame.pack();
        mainFrame.repaint();
    }

    // Method to create statistic panels (Before/After)
    private JPanel createStatisticPanel(String header, statisticSet stats) {
        JPanel panel = new JPanel();
        panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS));

        JLabel headerLabel = new JLabel(header);

        //String blockChangeStr = String.format(showPercent ? "%.3f%%" : "%.3f", blockChange);

        JLabel totalBlocks = new JLabel(String.valueOf(stats.numBlocks));
        JLabel smallBlocks = new JLabel(String.valueOf(stats.numSmallHBs));
        JLabel avgPointsPerBlock = new JLabel(String.format("%.3f", stats.averageHBSize));

        panel.add(headerLabel);
        addWhiteSpace(panel);
        panel.add(totalBlocks);
        panel.add(smallBlocks);
        addWhiteSpace(panel);
        panel.add(avgPointsPerBlock);

        for (double avg : stats.averageSizeByClass) {
            panel.add(new JLabel(String.format("%.3f", avg)));
        }

        adjustLabelFontAndPadding(panel, SIMP_PANELS_FONTSIZE);

        return panel;
    }

    // Method to create the difference panel
    private JPanel createDifferencePanel(statisticSet sBefore, statisticSet sAfter) {
        JPanel panel = new JPanel();
        panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS));

        // Calculate the differences
        double blockChange = ((double) (sAfter.numBlocks - sBefore.numBlocks) / sBefore.numBlocks) * 100.0;
        double smallBlockChange = ((double) (sAfter.numSmallHBs - sBefore.numSmallHBs) / sBefore.numSmallHBs) * 100.0;
        double averageBlockSizeChange = ((sAfter.averageHBSize - sBefore.averageHBSize) / sBefore.averageHBSize) * 100.0;

        // Handle raw mode
        if (!showPercent) {
            blockChange = sAfter.numBlocks - sBefore.numBlocks;
            smallBlockChange = sAfter.numSmallHBs - sBefore.numSmallHBs;
            averageBlockSizeChange = sAfter.averageHBSize - sBefore.averageHBSize;
        }

        // Format the numbers with percentage symbols in percent mode
        String blockChangeStr = String.format(showPercent ? "%s%.3f%%" : "%s%.3f", signOfNumber(blockChange), blockChange);
        String smallBlockChangeStr = String.format(showPercent ? "%s%.3f%%" : "%s%.3f",signOfNumber(smallBlockChange), smallBlockChange);
        String avgBlockSizeChangeStr = String.format(showPercent ? "%s%.3f%%" : "%s%.3f",signOfNumber(averageBlockSizeChange), averageBlockSizeChange);

        // Add components to the panel
        JLabel diffHeader = new JLabel("Difference");
        JLabel tb_Diff = new JLabel(blockChangeStr);
        JLabel sb_Diff = new JLabel(smallBlockChangeStr);
        JLabel avgPPB_Diff = new JLabel(avgBlockSizeChangeStr);


        // Apply color coding
        applyColorCoding(tb_Diff, blockChange);
        applyColorCoding(sb_Diff, smallBlockChange);
        applyColorCoding(avgPPB_Diff, averageBlockSizeChange);

        panel.add(diffHeader);
        addWhiteSpace(panel);
        panel.add(tb_Diff);
        panel.add(sb_Diff);
        addWhiteSpace(panel);
        panel.add(avgPPB_Diff);

        adjustLabelFontAndPadding(panel, SIMP_PANELS_FONTSIZE);
        return panel;
    }


    private char signOfNumber(double num) {
        return Math.signum(num) > 0 ? '+' : (Math.signum(num) < 0 ? '-' : '0');
    }
    /////


    /**
     * Helper function for color coding the difference columns.
     * neg = Red, pos = Green, 0 = Black.
     * @param label The label to apply the color to.
     * @param value The value the label is holding.
     */
    private void applyColorCoding(JLabel label, double value) {
        if (value > 0) {
            label.setForeground(new Color(70, 128, 66));
        } else if (value < 0) {
            label.setForeground(new Color(140, 22, 46));
        } else {
            label.setForeground(Color.BLACK);
        }
    }

    /**
     * Function to recursively increase the size of all labels that fall within a panel.
     * Will go to sub-panels and try to increase their label font size too.
     * @param panel JPanel to use.
     * @param size Font size.
     */
    private void adjustLabelFontAndPadding(JPanel panel, int size) {
        for (Component comp : panel.getComponents()) {
            if (comp instanceof JLabel) {
                JLabel label = (JLabel) comp;
                Font currentFont = label.getFont();
                label.setFont(new Font(currentFont.getName(), currentFont.getStyle(), size));
                label.setBorder(new EmptyBorder(0, 0, 10, 0));
            }
            else if (comp instanceof JPanel){
                adjustLabelFontAndPadding((JPanel) comp, size);
            }
        }

        // Revalidate and repaint the panel to reflect changes
        panel.revalidate();
        panel.repaint();
    }



    private JPanel createBlockPanel() {
        JPanel panel = new JPanel(new GridLayout(1, 4));
        panel.setPreferredSize(new Dimension(1600, 900));

        JPanel labelPanel = new JPanel();
        labelPanel.setLayout(new BoxLayout(labelPanel, BoxLayout.Y_AXIS));
        ArrayList<String> labels = new ArrayList<>(List.of("\n", "Total Blocks", "Small Blocks",
                                                            "Avg. Points per Block", "Total"));
        for(String uClass : DV.uniqueClasses){
            labels.add("Class \"" + uClass + "\"");
        }

        // If a label is a header, mark it. //TODO: find a better way to do this later.
        Set<Integer> headers = new HashSet<>(Set.of(3));

        addWhiteSpace(labelPanel);
        for (int i = 0; i < labels.size(); i++) {
            JLabel tmp = new JLabel(labels.get(i));

            // Check if the current index is a header
            if (headers.contains(i)) {
                tmp.setFont(tmp.getFont().deriveFont(Font.BOLD)); // Set font to bold
            }

            // Add the label to the panel
            labelPanel.add(tmp);
        }


        statisticSet sBefore = statisticHistory.get(firstSetIdx);
        statisticSet sAfter = statisticHistory.get(secondSetIdx);

        JPanel beforePanel = createStatisticPanel("Before", sBefore);
        JPanel afterPanel = createStatisticPanel("After", sAfter);


        JPanel differencePanel = createDifferencePanel(sBefore, sAfter);
        JButton toggleUnit = new JButton(showPercent ? "Show Raw Numbers" : "Show Percentages");
        toggleUnit.addActionListener(e -> {
            showPercent = !showPercent;

            // Update difference panel
            differencePanel.removeAll();
            JPanel updatedDifferencePanel = createDifferencePanel(sBefore, sAfter);
            for (Component comp : updatedDifferencePanel.getComponents()) {
                differencePanel.add(comp);
            }

            // Update toggle button text
            toggleUnit.setText(showPercent ? "Show Raw Numbers" : "Show Percentages");

            // Revalidate and repaint the container
            differencePanel.revalidate();
            differencePanel.repaint();
        });

        // Add the difference panel and toggle button
        panel.add(labelPanel);
        panel.add(beforePanel);
        panel.add(afterPanel);
        panel.add(differencePanel);
        panel.add(toggleUnit);

        adjustLabelFontAndPadding(panel, SIMP_PANELS_FONTSIZE);

        return panel;

    }



    private void addWhiteSpace(JPanel panel){
        panel.add(Box.createRigidArea(new Dimension(0, 20)));
    }

    private JPanel createClausePanel() {
        JPanel panel = new JPanel();
        panel.setPreferredSize(new Dimension(1600, 900));
        JLabel label = new JLabel("Clauses");
        panel.add(label);

        return panel;
    }

    private JScrollPane byBlockAttributeVis(statisticSet sBefore, statisticSet sAfter){
        // Create the main panel with a preferred size
        JPanel panel = new JPanel();
        panel.setPreferredSize(new Dimension(1600, 900));

        // Overall size needed for all the graphics
        int totalImages = DV.uniqueClasses.size() + 1; // Including an optional "overall" category
        int cols = 3;
        int rows = (int) Math.ceil((double) totalImages / cols);

        // Use GridLayout to arrange images
        panel.setLayout(new GridLayout(rows, cols, 10, 10)); // 10px gap between components

        // Calculate grid dimensions based on DV.fieldLen
        int cellSize = 50; // Cell dimensions
        int gridSize = (int) Math.ceil(Math.sqrt(DV.fieldLength)); // Determine grid rows/cols

        // Add a component for each block.
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
                JPanel graphicPanel = new JPanel(new BorderLayout());
                JLabel graphicLabel = new JLabel(String.format("Block: %d -- Class %s", uniqueId, DV.uniqueClasses.get(bClass)));

                // Extract attributes (excluding last two elements)
                Set<Integer> removedSet = new HashSet<>(beforeBlock.subList(0, beforeBlock.size() - 2));
                Set<Integer> beforeSet = new HashSet<>(beforeBlock.subList(0, beforeBlock.size() - 2));

                Set<Integer> afterAttributes = new HashSet<>(afterBlock.subList(0, afterBlock.size() - 2));

                // Find removed attributes (present in before but not in after)
                removedSet.removeAll(afterAttributes);

                // Generate the grid image
                BufferedImage gridImage = createGridImage(gridSize, gridSize, cellSize, cellSize, DV.fieldLength, removedSet, beforeSet);
                JLabel imageElement = new JLabel(new ImageIcon(gridImage));

                // Add components to the graphic panel
                graphicPanel.add(graphicLabel, BorderLayout.NORTH);
                graphicPanel.add(imageElement, BorderLayout.CENTER);

                // Add the graphic panel to the main panel
                panel.add(graphicPanel);
            }
        }

        // Add the panel to the JFrame and make it visible
        return new JScrollPane(panel);
    }

    private JScrollPane byClassAttributeVis(statisticSet sBefore, statisticSet sAfter){
        // Create the main panel with a preferred size
        JPanel panel = new JPanel();
        panel.setPreferredSize(new Dimension(1600, 900));

        // Overall size needed for all the graphics
        int totalImages = DV.uniqueClasses.size() + 1; // Including an optional "overall" category
        int cols = 3;
        int rows = (int) Math.ceil((double) totalImages / cols);

        // Use GridLayout to arrange images
        panel.setLayout(new GridLayout(rows, cols, 10, 10)); // 10px gap between components

        // Calculate grid dimensions based on DV.fieldLen
        int cellSize = 50; // Cell dimensions
        int gridSize = (int) Math.ceil(Math.sqrt(DV.fieldLength)); // Determine grid rows/cols

        // Add components for each class
        for (int i = 0; i < DV.uniqueClasses.size(); i++) {
            // Create a new panel for each graphic
            JPanel graphicPanel = new JPanel(new BorderLayout());
            JLabel graphicLabel = new JLabel("Class: " + DV.uniqueClasses.get(i));

            // Compute removed and existing attributes
            ArrayList<Integer> removed = new ArrayList<>(sBefore.usedAttributesByClass.get(i));
            removed.removeAll(sAfter.usedAttributesByClass.get(i));
            Set<Integer> removedSet = new HashSet<>(removed);
            Set<Integer> beforeSet = new HashSet<>(sBefore.usedAttributesByClass.get(i));

            // Generate the grid image
            BufferedImage gridImage = createGridImage(gridSize, gridSize, cellSize, cellSize, DV.fieldLength, removedSet, beforeSet);
            JLabel imageElement = new JLabel(new ImageIcon(gridImage));

            // Add components to the graphic panel
            graphicPanel.add(graphicLabel, BorderLayout.NORTH);
            graphicPanel.add(imageElement, BorderLayout.CENTER);

            // Add the graphic panel to the main panel
            panel.add(graphicPanel);
        }

        // Add the panel to the JFrame and make it visible
        return new JScrollPane(panel);
    }

    private void createAttributeVisWindow(statisticSet sBefore, statisticSet sAfter) {
        // Create a new JFrame for the attribute visualization
        attributeVisWindow = new JFrame("Attribute Visualization");
        attributeVisWindow.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);

        // Create the JTabbedPane
        JTabbedPane tabbedPane = new JTabbedPane();

        // Add tabs for each visualization
        JScrollPane byClass = byClassAttributeVis(sBefore, sAfter);
        tabbedPane.addTab("By Class", byClass);

        JScrollPane byBlock = byBlockAttributeVis(sBefore, sAfter);
        tabbedPane.addTab("By Block", byBlock);

        // Add the tabbed pane to the window
        attributeVisWindow.add(tabbedPane);

        attributeVisWindow.pack();
        attributeVisWindow.setLocationRelativeTo(null);
        attributeVisWindow.setVisible(false);
    }

    private JPanel createAttrPanel() {
        statisticSet sBefore = statisticHistory.get(firstSetIdx);
        statisticSet sAfter = statisticHistory.get(secondSetIdx);
        JPanel panel = new JPanel(new BorderLayout());

        createAttributeVisWindow(sBefore, sAfter);

        JButton showVisBtn = new JButton("Show Visualizations");
        showVisBtn.addActionListener(e -> attributeVisWindow.setVisible(true));
        panel.add(showVisBtn, BorderLayout.SOUTH);

        return panel;
    }




    // Method to create a grid as a BufferedImage
    private BufferedImage createGridImage(int rows, int cols, int cellWidth, int cellHeight, int totalAttributes, Set<Integer> removedSet, Set<Integer> beforeSet) {
        int imageWidth = cols * cellWidth;
        int imageHeight = rows * cellHeight;


        // Create a blank image
        BufferedImage image = new BufferedImage(imageWidth, imageHeight, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g2d = image.createGraphics();

        // Enable anti-aliasing for better rendering
        g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        int attributeNum = 0; // Attribute counter

        // Draw the grid
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                int x = col * cellWidth;
                int y = row * cellHeight;

                // Determine color based on attribute state
                if (attributeNum < totalAttributes) {
                    if (removedSet.contains(attributeNum)) {
                        g2d.setColor(Color.RED);
                    }else if(beforeSet.contains(attributeNum)) {
                        g2d.setColor(Color.GREEN);
                    }
                    else{
                        g2d.setColor(Color.GRAY);
                    }
                } else {
                    g2d.setColor(Color.BLACK);
                }

                // Fill the cell
                g2d.fillRect(x, y, cellWidth, cellHeight);

                // Draw the cell border
                g2d.setColor(Color.BLACK);
                g2d.drawRect(x, y, cellWidth, cellHeight);

                // Draw the cell text for valid attributes
                if (attributeNum < totalAttributes) {
                    g2d.setColor(Color.BLACK);
                    g2d.setFont(new Font("Arial", Font.BOLD, 12));
                    String text = String.format("x%d",attributeNum);
                    FontMetrics fm = g2d.getFontMetrics();

                    int textX = x + (cellWidth - fm.stringWidth(text)) / 2;
                    int textY = y + (cellHeight + fm.getAscent()) / 2 - 2;
                    g2d.drawString(text, textX, textY);
                }

                attributeNum++;
            }
        }

        g2d.dispose(); // Release graphics context
        return image;
    }


    private JPanel createSimplificationPanel() {
        JPanel panel = new JPanel(new BorderLayout());
        panel.setPreferredSize(new Dimension(1600, 900));
        JLabel label = new JLabel("Simplifications");
        panel.add(label);

        // Add the list of simplifications
        JScrollPane simpScroll = new JScrollPane();
        panel.add(simpScroll, BorderLayout.LINE_START);


        int last = statisticHistory.size() - 1;
        JPanel simpList = new JPanel();
        simpList.setLayout(new BoxLayout(simpList, BoxLayout.Y_AXIS));


        JLabel temp = new JLabel(" 0. None");
        temp.setFont(new Font("Tahoma", Font.PLAIN, 14));
        temp.setBorder(BorderFactory.createEmptyBorder(5, 0, 5, 0));
        simpList.add(temp);
        simpList.add(Box.createVerticalStrut(10));

        for(int i = 0; i < statisticHistory.get(last).algoLog.size(); i++){

            String simp = statisticHistory.get(last).algoLog.get(i);
            JLabel elementLabel = new JLabel(" " + (i+1) + ". " + simp + "  ");
            elementLabel.setFont(new Font("Tahoma", Font.PLAIN, 14));
            elementLabel.setBorder(BorderFactory.createEmptyBorder(5, 0, 5, 0));

            simpList.add(elementLabel);
            simpList.add(Box.createVerticalStrut(10));
        }

        simpList.setBackground(new Color(219, 218, 215));
        simpList.setBorder(BorderFactory.createLineBorder(new Color(219, 218, 215), 1));

        simpScroll.setViewportView(simpList);

        // Add the comparison picker portion
        JPanel comparePicker = new JPanel();
        panel.add(comparePicker, BorderLayout.CENTER);


        // Allow user to select the indices of the sets they want to compare
        // first and second spinner should be limited to 0 + range
        JLabel compareLbl = new JLabel("");
        JSpinner firstSpinner = new JSpinner();
        JTextField firstSetTxtFld = new JTextField("None");


        JLabel toLbl = new JLabel("To");
        JSpinner secondSpinner = new JSpinner();
        JTextField secondSetTxtFld = new JTextField("");

        SpinnerNumberModel firstSpinnerModel = new SpinnerNumberModel(0, 0, statisticHistory.size() - 1, 1);
        firstSpinner.setModel(firstSpinnerModel);

        SpinnerNumberModel secondSpinnerModel = new SpinnerNumberModel(statisticHistory.size() - 1, 0, statisticHistory.size() - 1, 1);
        secondSpinner.setModel(secondSpinnerModel);

        JButton generateComparison = new JButton("Generate Comparison");
        generateComparison.addActionListener(e->{
            firstSetIdx = (Integer) firstSpinner.getValue();
            secondSetIdx = (Integer) secondSpinner.getValue();

            if (firstSetIdx == secondSetIdx || secondSetIdx < firstSetIdx) {
                JOptionPane.showMessageDialog(null, "Please select different indices for comparison.");
                return;
            }
            System.out.printf("Comparing %d set to %d set", firstSetIdx, secondSetIdx);

            // Compare the stat set at elements
            genGUIComparison();

        });


        comparePicker.add(compareLbl);
        comparePicker.add(firstSpinner);
        comparePicker.add(firstSetTxtFld);
        comparePicker.add(toLbl);
        comparePicker.add(secondSpinner);
        comparePicker.add(secondSetTxtFld);
        comparePicker.add(generateComparison);
        return panel;
    }

    private void genGUIComparison(){
       attrPanel = createAttrPanel();
       blockPanel = createBlockPanel();
       clausePanel = createClausePanel();
    }


    private JToolBar createToolBar(){
        JToolBar toolBar = new JToolBar();
        toolBar.setFloatable(false);
        toolBar.setLayout(new FlowLayout(FlowLayout.LEFT));

        JButton simpTabBtn = new JButton("Simplifications Done");
        simpTabBtn.setPreferredSize(new Dimension(200, 25));
        simpTabBtn.addActionListener(e ->{
            mainFrame.getContentPane().remove(mainFrame.getContentPane().getComponent(2));
            mainFrame.add(simpPanel, BorderLayout.CENTER);
            mainFrame.revalidate();
            mainFrame.repaint();
        });

        JButton clauseTabBtn= new JButton("Clauses");
        clauseTabBtn.setPreferredSize(new Dimension(200, 25));
        clauseTabBtn.addActionListener(e -> navigationHandler(clausePanel));

        JButton blocksBtn = new JButton("Blocks");
        blocksBtn.setPreferredSize(new Dimension(200, 25));
        blocksBtn.addActionListener(e -> navigationHandler(blockPanel));

        JButton attrBtn = new JButton("Attributes");
        attrBtn.setPreferredSize(new Dimension(200, 25));
        attrBtn.addActionListener(e -> navigationHandler(attrPanel));


        toolBar.add(simpTabBtn);
        toolBar.add(clauseTabBtn);
        toolBar.add(blocksBtn);
        toolBar.add(attrBtn);

        return toolBar;
    }

    /**
     * Shows the panel for a statistics panel. If a comparison is generated, we will
     * show the page the button clicked on.
     * @param panelToShow The comparison panel to show. Ex. clausePanel
     */
    private void navigationHandler(JPanel panelToShow){
        if (firstSetIdx != -1 && secondSetIdx != -1) {
            // Show the attributes tab
            mainFrame.getContentPane().remove(mainFrame.getContentPane().getComponent(2));
            mainFrame.add(panelToShow, BorderLayout.CENTER);
            mainFrame.revalidate();
            mainFrame.repaint();
        }
        else{
            JOptionPane.showMessageDialog(mainFrame,
                    "Please choose two block states to compare, then click 'Generate Statistics'.",
                    "Selection Required",
                    JOptionPane.WARNING_MESSAGE);
        }
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
        if (console) {
            Scanner scan = new Scanner(System.in);
            System.out.println("Before: ");
            before = scan.nextInt();
            System.out.println("After: ");
            after = scan.nextInt();


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
            } else {
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
                System.out.printf("\n\tBlock #%d from class \"%s\" :", sBefore.usedAttributesPerBlock.get(i).get(n - 1), DV.uniqueClasses.get(last));

                // Condensed print all but last 2 elements of each block row.
                printIntervalCondensed(sBefore.usedAttributesPerBlock.get(i).subList(0, n - 1));
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
        else{
            statisticsGUI();
        }
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
                if ((float)data[i] >= tempBlock.minimums.get(i).get(j).floatValue() && (float) data[i] <= tempBlock.maximums.get(i).get(j).floatValue()) {
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

        return hbList;
    }
}
