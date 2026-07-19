function runBatchInspection(ds)
    % Batch image classifier
    
    % retrieve test set size
    numFiles = numel(ds.Files);
    
    % 1. Initialize arrays to store true and predicted labels
    trueLabels = categorical(NaN(1, 255)); 
    predictedLabels = categorical(NaN(1, 255));
    
    % 2. Batch Processing Loop
    for i = 1:numFiles
    
        % Load or generate test data for the current batch
        img = readimage(ds, i);
    
        % Extract ground truth and pass the data to the tubeClassifierNet model
        actual = ds.Labels(i);
        imgData = singleImageInspectionFunction(img, false, true);
    
        % Append results to your master lists
        trueLabels(i) = actual;
        predictedLabels(i) = imgData.AIClass;
    
    end
    
    % 3. Generate Confusion Chart
    figure;
    cm = confusionchart(trueLabels, predictedLabels);
    cm.Title = 'Batch Test Confusion Matrix';
    cm.RowDisplayLabels = {'Color Defect', 'Length Defect', 'Good', 'Metal Defect'};
    cm.ColumnDisplayLabels = {'Color D.', 'Length D.', 'Good', 'Metal D.'};
    cm.RowSummary = 'row-normalized'; % Shows class-specific true positive rates
    
    % true = pass, false = fail
    truePF = trueLabels == 'good';
    predictedPF = predictedLabels == 'good';
    
    figure;
    cmPF = confusionchart(truePF, predictedPF);
    cmPF.Title = 'Batch Test Confusion Matrix PF';
    cmPF.RowDisplayLabels = {'Pass', 'Fail'};
    cmPF.ColumnDisplayLabels = {'Pass', 'Fail'};
    
    % Yield Rates (Actual + Predicted)
    trueYield = sum(truePF) / numFiles * 100;
    predictedYield = sum(predictedPF) / numFiles * 100;
    out = sprintf("Actual Yield: %.1f%%\nPredicted Yield %.1f%%\n", ...
        trueYield, predictedYield);
    disp(out);
    
    
    % Defect Rates 
    trueDR = 100 - trueYield; 
    escapedDefects = (predictedPF == true & truePF == false);
    escapedDR = sum(escapedDefects); % Percentage of defects missed by the detection system
    out = sprintf("Defect Rate: %.1f%%\nEscaped Defect Rate %.1f%%\n", ...
        trueDR, escapedDR);
    disp(out);
end