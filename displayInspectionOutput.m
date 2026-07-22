function displayInspectionOutput(originalImg, reportData)
% colors
grey = [0.12 0.12 0.12]

% Create a figure sized moderately to render well in the Live Editor output
fig = figure('Name', 'Inspection Results', 'Color', grey, ...
    'Units', 'normalized', 'Position', [0.1 0.1 0.7 0.6]);

% Use a 2-row, 3-column layout to stack images while keeping text on the left
t = tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

% =====================================================================
% LEFT SIDE: TEXT & DECISION DASHBOARD (Spans 2 rows, 1 column)
% =====================================================================
axText = nexttile(1, [2 1]);
axis(axText, 'off'); % Hide the standard graph axes

% PASS/FAIL Giant Text 
if strcmp(reportData.Decision, 'PASS')
    decisionColor = [0 0.8 0]; % Bright Green
else
    decisionColor = [0.8 0 0]; % Bright Red
end

text(axText, 0.5, 0.85, reportData.Decision, 'FontSize', 40, ...
    'FontWeight', 'bold', 'Color', decisionColor, 'HorizontalAlignment', 'center');

% AI Classification Text
displayClass = strrep(reportData.AIClass, '_', ' '); 
aiClassText = sprintf('AI Classification:\n%s', displayClass);

text(axText, 0.5, 0.65, aiClassText, 'FontSize', 14, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'center');

% Vision Evidence Box 
evidenceText = sprintf(['--- Vision Evidence ---\n', ...
    'Confidence Score: %.2f%%\n', ...
    'Total Tubes: %d\n', ...
    'Normal Tubes: %d\n', ...
    'Shape Anomalies: %d\n', ...
    'Color Anomalies: %d'], ...
    reportData.Confidence, ...
    reportData.metrics.TotalTubes, reportData.metrics.NormalTubes, ...
    reportData.metrics.ShapeAnomalies, reportData.metrics.ColorAnomalies);

text(axText, 0.5, 0.25, evidenceText, 'FontSize', 12, 'HorizontalAlignment', 'center', ...
    'BackgroundColor', grey, 'EdgeColor', 'black', 'Margin', 5);

% =====================================================================
% RIGHT SIDE (TOP ROW): ORIGINAL & MASK IMAGES
% =====================================================================
% Image 1: Original
ax1 = nexttile(2);
imshow(originalImg, 'Parent', ax1);
title(ax1, 'Original Image', 'FontSize', 12);


% Image 2: Mask
ax2 = nexttile(3);
imshow(reportData.images.ImageMask, 'Parent', ax2);
title(ax2, 'Cleaned Mask', 'FontSize', 12);

% =====================================================================
% RIGHT SIDE (BOTTOM ROW): ANNOTATED IMAGE
% =====================================================================
% Image 3: Object Detection (Spans the bottom 2 columns)
ax3 = nexttile(5, [1 2]);
imshow(reportData.images.AnnotatedImage, 'Parent', ax3);
title(ax3, 'Object Detection', 'FontSize', 12);

% Main Title
title(t, 'Automated Tube Inspection Dashboard', 'FontSize', 18, 'FontWeight', 'bold');
end