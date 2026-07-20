function [numColorAnomalies, colorStats] = colorAnomalyDetection(img)

hsvImg = rgb2hsv(img);
hChannel = hsvImg(:,:,1);
sChannel = hsvImg(:,:,2);

% "Blue Tube" roughly means: Hue between 0.5-0.7, and high saturation
blueMask = (hChannel > 0.5) & (hChannel < 0.7) & (sChannel > 0.4);

% Clean up the mask to remove tiny blue glares or noise
blueMask = bwareaopen(blueMask, 500);
blueMask = imfill(blueMask, 'holes');

% Measure the colored tubes
colorStats = regionprops(blueMask, 'BoundingBox', 'Area');

% Track the number of color anomalies we found by counting the number of
% rows in the table. 
% We can ignore this section after we get this information.
numColorAnomalies = length(colorStats);

end