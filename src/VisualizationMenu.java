import javax.swing.*;
import javax.swing.filechooser.FileNameExtensionFilter;
import java.awt.*;
import java.awt.event.*;
import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class VisualizationMenu extends JPanel
{
    /**
     * Creates Visualization Options Menu on mouseLocation
     */
    public VisualizationMenu()
    {
        // visualization panel
        JPanel visPanel = new JPanel();
        visPanel.setLayout(new GridBagLayout());
        GridBagConstraints constraints = new GridBagConstraints();

        // choose plot type
        JButton plotBtn = new JButton("Plot Type");
        plotBtn.setToolTipText("Choose GLC-L or DSC2 plot");
        plotBtn.setFont(plotBtn.getFont().deriveFont(12f));
        plotBtn.addActionListener(e->
        {
            // radio button group
            ButtonGroup plotType = new ButtonGroup();
            JRadioButton glc = new JRadioButton("GLC-L", DV.glc_or_dsc);
            JRadioButton dsc = new JRadioButton("DSC2", DV.glc_or_dsc);
            plotType.add(glc);
            plotType.add(dsc);

            // default function panel
            JPanel plotTypePanel = new JPanel();
            plotTypePanel.add(new JLabel("Built-In: "));
            plotTypePanel.add(glc);
            plotTypePanel.add(dsc);

            JOptionPane.showConfirmDialog(DV.mainFrame, plotTypePanel, "Plot Type", JOptionPane.OK_CANCEL_OPTION, JOptionPane.PLAIN_MESSAGE);

            DV.glc_or_dsc = glc.isSelected();

            if (DV.data != null)
            {
                DataVisualization.optimizeSetup();
                DataVisualization.drawGraphs();
            }
        });

        constraints.gridx = 0;
        constraints.gridy = 0;
        constraints.fill = GridBagConstraints.HORIZONTAL;
        constraints.insets = new Insets(5, 5, 5, 5);
        visPanel.add(plotBtn, constraints);

        // choose class to visualize as main
        JButton chooseUpperClassBtn = new JButton("Upper Class");
        chooseUpperClassBtn.setToolTipText("Choose class to be visualized on upper graph");
        chooseUpperClassBtn.setFont(chooseUpperClassBtn.getFont().deriveFont(12f));
        chooseUpperClassBtn.addActionListener(e ->
        {
            if (DV.data != null)
            {
                int chosen = JOptionPane.showOptionDialog(
                        DV.mainFrame,
                        "Choose upper class.\nThe upper class will be visualized on the upper graph.",
                        "Choose Upper Class",
                        JOptionPane.YES_NO_CANCEL_OPTION,
                        JOptionPane.PLAIN_MESSAGE,
                        null,
                        DV.uniqueClasses.toArray(),
                        null);

                if (chosen != -1)
                {
                    // remove past accuracies and classifications
                    DV.prevAllDataCM.clear();
                    DV.prevAllDataClassifications.clear();

                    // set upper class
                    DV.upperClass = chosen;

                    // lower class gets all others
                    for (int i = 0; i < DV.classNumber; i++)
                    {
                        if (i != chosen)
                            DV.lowerClasses.set(i, true);
                        else
                            DV.lowerClasses.set(i, false);
                    }

                    // optimize setup then draw graphs
                    DataVisualization.optimizeSetup();
                    DataVisualization.drawGraphs();
                }
            }
            else
            {
                JOptionPane.showMessageDialog(
                        DV.mainFrame,
                        "Please create a project before choosing the upper class.\nFor additional information, please view the \"Help\" tab.",
                        "Error: not data",
                        JOptionPane.ERROR_MESSAGE);
            }
        });

        constraints.gridx = 1;
        constraints.gridy = 0;
        visPanel.add(chooseUpperClassBtn, constraints);

        // specify visualization
        JButton specifyVisBtn = new JButton("Specify Visualization");
        specifyVisBtn.setToolTipText("Removes one class from the lower graph");
        specifyVisBtn.setFont(specifyVisBtn.getFont().deriveFont(12f));
        specifyVisBtn.addActionListener(e ->
        {
            if (DV.data != null)
            {
                // classes on lower graph
                ArrayList<String> removableClasses = new ArrayList<>();
                ArrayList<String> classes = new ArrayList<>(DV.uniqueClasses);

                for (int i = 0; i < DV.classNumber; i++)
                {
                    if (DV.lowerClasses.get(i))
                        removableClasses.add(classes.get(i));
                }

                if (removableClasses.size() > 1)
                {
                    JLabel removableLabel = new JLabel("Removable Classes");
                    JComboBox<Object> removableList = new JComboBox<>(removableClasses.toArray());
                    removableList.setSelectedIndex(0);
                    removableList.setEditable(true);

                    JPanel removablePanel = new JPanel();
                    removablePanel.add(removableLabel);
                    removablePanel.add(removableList);

                    int choice = JOptionPane.showConfirmDialog(
                            DV.mainFrame,
                            removablePanel,
                            "Remove Class",
                            JOptionPane.OK_CANCEL_OPTION);

                    if (choice == 0)
                    {
                        // add previous analytics
                        DV.prevAllDataCM.add(DV.allDataCM);
                        DV.prevAllDataClassifications.add(DV.allDataClassifications);

                        // get class to be removed
                        int selected = removableList.getSelectedIndex();
                        String className = removableClasses.get(selected);

                        // remove class
                        for (int i = 0; i < DV.classNumber; i++)
                        {
                            if (className.equals(classes.get(i)))
                                DV.lowerClasses.set(i, false);
                        }

                        // optimize setup then draw graphs
                        DataVisualization.optimizeSetup();
                        DataVisualization.drawGraphs();
                    }
                }
                else
                    JOptionPane.showMessageDialog(DV.mainFrame, "Classes cannot be further separated.");
            }
            else
            {
                JOptionPane.showMessageDialog(
                        DV.mainFrame,
                        "Please create a project before specifying the visualization.\nFor additional information, please view the \"Help\" tab.",
                        "Error: not data",
                        JOptionPane.ERROR_MESSAGE);
            }
        });

        constraints.gridx = 0;
        constraints.gridy = 1;
        visPanel.add(specifyVisBtn, constraints);

        // visualize overlap area
        JButton visOverlapBtn = new JButton("Visualize Overlap");
        visOverlapBtn.setToolTipText("Visualize the overlap area");
        visOverlapBtn.setFont(visOverlapBtn.getFont().deriveFont(12f));
        visOverlapBtn.addActionListener(e ->
        {
            if (DV.classNumber > 1 && DV.accuracy < 100)
            {
                DV.drawOverlap = true;
                DataVisualization.drawGraphs();
            }
            else
                JOptionPane.showMessageDialog(DV.mainFrame, "No overlap area");
        });

        // stop visualizing overlap
        JButton stopOverlapVisBtn = new JButton("Stop Visualizing Overlap");
        stopOverlapVisBtn.setToolTipText("Visualize all data");
        stopOverlapVisBtn.setFont(stopOverlapVisBtn.getFont().deriveFont(12f));
        stopOverlapVisBtn.addActionListener(e ->
        {
            DV.drawOverlap = false;
            DataVisualization.drawGraphs();
        });

        constraints.gridx = 1;
        constraints.gridy = 1;

        if (!DV.drawOverlap)
            visPanel.add(visOverlapBtn, constraints);
        else
            visPanel.add(stopOverlapVisBtn, constraints);

        // change visualization function for each attribute of each vector
        JButton scalarVisFuncBtn = new JButton("Scalar Function");
        scalarVisFuncBtn.setToolTipText("Applies given function to all attributes of all data points");
        scalarVisFuncBtn.setFont(scalarVisFuncBtn.getFont().deriveFont(12f));
        scalarVisFuncBtn.addActionListener(e ->
        {
            // popup asking for number of folds
            JPanel funcPanel = new JPanel(new BorderLayout());

            // text panel
            JPanel textPanel = new JPanel(new FlowLayout(FlowLayout.CENTER));

            // maximum text field
            JTextField funcField = new JTextField();
            funcField.setPreferredSize(new Dimension(200, 30));
            funcField.setText(DV.scalarFunction);
            JLabel funcLabel = new JLabel("Function: f(x) = ");
            funcLabel.setFont(funcLabel.getFont().deriveFont(12f));
            textPanel.add(funcLabel);
            textPanel.add(funcField);

            // add text panel
            funcPanel.add(textPanel, BorderLayout.SOUTH);

            Object[] funcButtons = { "Ok", "Cancel", "Help" };

            boolean notChosen = true;

            // loop until folds are valid or user quits
            while (notChosen)
            {
                int choice = JOptionPane.showOptionDialog(
                        DV.mainFrame, funcPanel,
                        "Enter function",
                        JOptionPane.YES_NO_CANCEL_OPTION,
                        JOptionPane.PLAIN_MESSAGE,
                        null,
                        funcButtons,
                        funcButtons[0]);

                switch (choice)
                {
                    case 0 ->
                    {
                        // get function and remove spaces
                        String func = funcField.getText();

                        try
                        {
                            // try new function with dummy variables
                            Map<String, Double> variables = new HashMap<>();
                            FunctionParser.Expression exp = FunctionParser.parseScalerExpression(func, variables);

                            variables.put("x", 0.5);
                            exp.eval();

                            // apply function if working
                            DV.scalarFunction = func;

                            for (int i = 0; i < DV.normalizedData.size(); i++)
                            {
                                for (int j = 0; j < DV.normalizedData.get(i).data.length; j++)
                                {
                                    for (int k = 0; k < DV.fieldLength; k++)
                                    {
                                        variables.put("x", DV.data.get(i).data[j][k]);
                                        DV.data.get(i).data[j][k] = exp.eval();
                                    }
                                }
                            }

                            DV.crossValidationNotGenerated = true;
                            DataVisualization.optimizeSetup();
                            DataVisualization.drawGraphs();
                        }
                        catch (Exception exc)
                        {
                            // invalid function input
                            JOptionPane.showMessageDialog(
                                    DV.mainFrame,
                                    """
                                            Error: input is invalid.
                                            Please enter a valid function.
                                            Select "Help" for more info.
                                            """,
                                    "Error",
                                    JOptionPane.ERROR_MESSAGE);
                        }

                        notChosen = false;
                    }
                    case 2 -> DV.scalarFuncInfoPopup();
                    default -> { return; }
                }
            }
        });

        constraints.gridx = 0;
        constraints.gridy = 2;
        visPanel.add(scalarVisFuncBtn, constraints);

        /**
         * START CONSTRUCTION ZONE
         */
        // change visualization function for each vector
        JButton vectorVisFuncBtn = new JButton("n-D Point Function");
        vectorVisFuncBtn.setToolTipText("Applies given function to all n-D points");
        vectorVisFuncBtn.setFont(vectorVisFuncBtn.getFont().deriveFont(12f));
        vectorVisFuncBtn.addActionListener(e ->
        {
            // popup asking for number of folds
            JPanel funcPanel = new JPanel();
            funcPanel.setLayout(new BoxLayout(funcPanel, BoxLayout.Y_AXIS));

            // svm vectors or input vectors
            ButtonGroup vectorType = new ButtonGroup();
            JRadioButton svmVec = new JRadioButton("SVM Vectors", true);
            JRadioButton userVec = new JRadioButton("User n-D Points");
            vectorType.add(svmVec);
            vectorType.add(userVec);

            // vector type panel
            JPanel vecPanel = new JPanel();
            JLabel vecLabel = new JLabel("n-D Point Type: ");
            vecLabel.setFont(vecLabel.getFont().deriveFont(12f));
            vecPanel.add(vecLabel);
            vecPanel.add(svmVec);
            vecPanel.add(userVec);
            JButton userVecInput = new JButton("User n-D Points Input");
            userVecInput.addActionListener(ee ->
            {
                if (DV.data.size() > 0)
                {
                    // set filter on file chooser
                    JFileChooser fileDialog = new JFileChooser();
                    fileDialog.setFileFilter(new FileNameExtensionFilter("csv", "csv"));

                    // set to current directory
                    File workingDirectory = new File(System.getProperty("user.dir"));
                    fileDialog.setCurrentDirectory(workingDirectory);

                    // open file dialog
                    int results = fileDialog.showOpenDialog(DV.mainFrame);

                    if (results == JFileChooser.APPROVE_OPTION)
                    {
                        File importFile = fileDialog.getSelectedFile();

                        // check if import was successful
                        boolean success = DataSetup.setupImportData(importFile);

                        // create graphs
                        if (success)
                        {
                            DataSetup.setupSupportVectors(importFile);
                        }
                        else
                        {
                            // add blank graph
                            JOptionPane.showMessageDialog(
                                    DV.mainFrame,
                                    "Please ensure the file is properly formatted.\nFor additional information, please view the \"Help\" tab.",
                                    "Error: could not open file",
                                    JOptionPane.ERROR_MESSAGE);
                        }
                    }
                    else if (results != JFileChooser.CANCEL_OPTION)
                    {
                        JOptionPane.showMessageDialog(
                                DV.mainFrame,
                                "Please ensure the file is properly formatted.\nFor additional information, please view the \"Help\" tab.",
                                "Error: could not open file",
                                JOptionPane.ERROR_MESSAGE);
                    }
                }
                else
                {
                    JOptionPane.showMessageDialog(
                            DV.mainFrame,
                            "Please create a project before importing data.\nFor additional information, please view the \"Help\" tab.",
                            "Error: could not import data",
                            JOptionPane.ERROR_MESSAGE);
                }

                // repaint and revalidate graph
                DV.graphPanel.repaint();
                DV.graphPanel.revalidate();
            });
            vecPanel.add(userVecInput);
            funcPanel.add(vecPanel);

            // radio button group
            ButtonGroup stockFunc = new ButtonGroup();
            JRadioButton svmPolyFunc = new JRadioButton("SVM - Polynomial Kernel");
            JRadioButton svmRBFFunc = new JRadioButton("SVM - RBF Kernel");
            JRadioButton customFunc = new JRadioButton("Custom");
            JRadioButton noFunc = new JRadioButton("None");
            stockFunc.add(svmPolyFunc);
            stockFunc.add(svmRBFFunc);
            stockFunc.add(customFunc);
            stockFunc.add(noFunc);

            // default function panel
            JPanel stockPanel = new JPanel();
            JLabel stockLabel = new JLabel("Built-In: ");
            stockLabel.setFont(stockLabel.getFont().deriveFont(12f));
            stockPanel.add(stockLabel);
            stockPanel.add(svmPolyFunc);
            stockPanel.add(svmRBFFunc);
            stockPanel.add(customFunc);
            stockPanel.add(noFunc);
            funcPanel.add(stockPanel);

            // text panel
            JPanel textPanel = new JPanel(new FlowLayout(FlowLayout.CENTER));

            // maximum text field
            JTextField funcField = new JTextField();
            funcField.setPreferredSize(new Dimension(200, 30));
            funcField.setText(DV.vectorFunction);
            JLabel funcLabel = new JLabel("Function: f(x, y) = ");
            funcLabel.setFont(funcLabel.getFont().deriveFont(12f));
            textPanel.add(funcLabel);
            textPanel.add(funcField);

            // set text
            if (DV.vectorFunction.equals("(1/" + DV.standardFieldLength + " * dot(x, y) + 1)^3"))
                svmPolyFunc.setSelected(true);
            else if (DV.vectorFunction.equals("e^(-1/" + DV.standardFieldLength + " * norm(vSub(x, y))^2)"))
                svmRBFFunc.setSelected(true);
            else if (DV.vectorFunction.equals("N/A"))
                noFunc.setSelected(true);
            else
                customFunc.setSelected(true);

            // add text panel
            funcPanel.add(textPanel);

            // add listeners
            svmPolyFunc.addActionListener(e1 -> funcField.setText("(1/" + DV.standardFieldLength + " * dot(x, y) + 1)^3"));
            svmRBFFunc.addActionListener(e1 -> funcField.setText("e^(-1/" + DV.standardFieldLength + " * norm(vSub(x, y))^2)"));
            noFunc.addActionListener(e1 -> funcField.setText("N/A"));

            funcField.addKeyListener(new KeyListener()
            {
                @Override
                public void keyTyped(KeyEvent e)
                {
                    customFunc.setSelected(true);
                }

                @Override
                public void keyPressed(KeyEvent e) {}

                @Override
                public void keyReleased(KeyEvent e) {}
            });

            Object[] funcButtons = { "Ok", "Cancel", "Help" };

            boolean notChosen = true;

            // loop until folds are valid or user quits
            while (notChosen)
            {
                int choice = JOptionPane.showOptionDialog(
                        DV.mainFrame, funcPanel,
                        "Enter function",
                        JOptionPane.YES_NO_CANCEL_OPTION,
                        JOptionPane.PLAIN_MESSAGE,
                        null,
                        funcButtons,
                        funcButtons[0]);

                switch (choice)
                {
                    case 0 ->
                    {
                        // get function and remove spaces
                        String func = funcField.getText();

                        if (func.equals("N/A"))
                        {
                            DV.fieldLength = DV.standardFieldLength;

                            DV.fieldNames = new ArrayList<>();
                            DV.fieldNames.addAll(DV.standardFieldNames);

                            DV.data = new ArrayList<>();
                            DV.data.addAll(DV.normalizedData);

                            DV.crossValidationNotGenerated = true;
                            DV.angleSliderPanel.setPreferredSize(new Dimension(Resolutions.angleSliderPanel[0], (100 * DV.fieldLength)));

                            DataVisualization.optimizeSetup();
                            DataVisualization.drawGraphs();
                        }
                        else
                        {
                            try
                            {
                                // try new function with dummy variables
                                Map<String, FunctionParser.VectorExpression> variables = new HashMap<>();
                                FunctionParser.Expression exp = FunctionParser.parseVectorExpression(func, variables);

                                variables.put("x", () -> new double[]{0, 0.3, 0.8});
                                variables.put("y", () -> new double[]{1, 0.6, 0.1});
                                exp.eval();

                                // apply function if working
                                DV.vectorFunction = func;
                                ArrayList<double[][]> splitByClass = new ArrayList<>();

                                for (int i = 0; i < DV.normalizedData.size(); i++)
                                {
                                    ArrayList<double[]> classData = new ArrayList<>();

                                    for (int j = 0; j < DV.normalizedData.get(i).data.length; j++)
                                    {
                                        ArrayList<Double> newRow = new ArrayList<>();

                                        for (int k = 0; k < DV.supportVectors.data.length; k++)
                                        {
                                            final double[] x = DV.normalizedData.get(i).data[j];
                                            final double[] y = DV.supportVectors.data[k];

                                            variables.put("x", () -> x);
                                            variables.put("y", () -> y);
                                            newRow.add(exp.eval());
                                        }

                                        double[] newRowArray = new double[newRow.size()];

                                        for (int w = 0; w < newRow.size(); w++)
                                            newRowArray[w] = newRow.get(w);

                                        classData.add(newRowArray);
                                    }

                                    double[][] newClassData = new double[classData.size()][];

                                    for (int w = 0; w < classData.size(); w++)
                                        newClassData[w] = classData.get(w);

                                    splitByClass.add(newClassData);
                                }

                                DV.data = DataSetup.createDataObjects(splitByClass);

                                DV.fieldLength = splitByClass.get(0)[0].length;

                                DV.angles = new double[DV.fieldLength];
                                DV.prevAngles = new double[DV.fieldLength];
                                DV.fieldNames.clear();

                                for (int i = 0; i < DV.fieldLength; i++)
                                {
                                    DV.fieldNames.add("feature " + i);
                                    DV.angles[i] = 45;
                                    DV.prevAngles[i] = 45;
                                }

                                DV.crossValidationNotGenerated = true;
                                DV.angleSliderPanel.setPreferredSize(new Dimension(Resolutions.angleSliderPanel[0], (100 * DV.fieldLength)));

                                DataVisualization.optimizeSetup();
                                DataVisualization.drawGraphs();
                            }
                            catch (Exception exc)
                            {
                                // invalid function input
                                JOptionPane.showMessageDialog(
                                        DV.mainFrame,
                                        """
                                                Error: input is invalid.
                                                Please enter a valid function.
                                                Select "Help" for more info.
                                                """,
                                        "Error",
                                        JOptionPane.ERROR_MESSAGE);
                            }
                        }

                        notChosen = false;
                    }
                    case 2 -> DV.vectorFuncInfoPopup();
                    default -> { return; }
                }
            }
        });

        constraints.gridx = 1;
        constraints.gridy = 2;
        visPanel.add(vectorVisFuncBtn, constraints);
        /**
         * END CONSTRUCTION ZONE
         */

        JCheckBox domainActiveBox = new JCheckBox("Domain Active", DV.domainActive);
        domainActiveBox.setToolTipText("Whether the domain is active or not.");
        domainActiveBox.setFont(domainActiveBox.getFont().deriveFont(12f));
        domainActiveBox.addActionListener(eee ->
        {
            DV.domainActive = domainActiveBox.isSelected();
            if (DV.data != null)
                DataVisualization.drawGraphs();
        });

        constraints.gridx = 0;
        constraints.gridy = 3;
        visPanel.add(domainActiveBox, constraints);

        JCheckBox drawFirstLineBox = new JCheckBox("First Line", DV.showFirstSeg);
        drawFirstLineBox.setToolTipText("Whether to draw the first line segment of a graph or not.");
        drawFirstLineBox.setFont(drawFirstLineBox.getFont().deriveFont(12f));
        drawFirstLineBox.addActionListener(fle ->
        {
            DV.showFirstSeg = drawFirstLineBox.isSelected();
            if (DV.data != null)
                DataVisualization.drawGraphs();
        });

        constraints.gridx = 1;
        constraints.gridy = 3;
        visPanel.add(drawFirstLineBox, constraints);

        // open analytics in another window
        JButton separateVisBtn = new JButton("Visualization Window");
        separateVisBtn.setToolTipText("Open another window displaying the visualization");
        separateVisBtn.setFont(separateVisBtn.getFont().deriveFont(12f));
        separateVisBtn.addActionListener(e->
        {
            if (!DV.displayRemoteGraphs)
            {
                DV.displayRemoteGraphs = true;

                JOptionPane optionPane = new JOptionPane(DV.remoteGraphPanel, JOptionPane.PLAIN_MESSAGE, JOptionPane.DEFAULT_OPTION, null, new Object[]{}, null);
                JDialog dialog = optionPane.createDialog(DV.mainFrame, "Visualization");
                dialog.setModal(false);
                dialog.setVisible(true);

                dialog.addWindowListener(new WindowAdapter()
                {
                    @Override
                    public void windowClosing(WindowEvent e)
                    {
                        DV.displayRemoteGraphs = false;
                        DV.remoteGraphPanel.removeAll();
                    }
                });

                if (DV.data != null)
                    DataVisualization.drawGraphs();
            }
        });

        constraints.gridx = 0;
        constraints.gridy = 4;
        constraints.gridwidth = 2;
        visPanel.add(separateVisBtn, constraints);

        JOptionPane.showOptionDialog(DV.mainFrame, visPanel, "Analytics Options", JOptionPane.DEFAULT_OPTION, JOptionPane.PLAIN_MESSAGE, null, new Object[]{}, null);
    }
}