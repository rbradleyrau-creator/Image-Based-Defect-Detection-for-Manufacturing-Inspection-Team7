function [numNormalTubes, numShapeAnomalies, totalTubesInImage, shapeMask, annotatedImage] = shapeAnomalyDetection(img, colorStats, numColorAnomalies)

gs = im2gray(img);
    
    % Apply a medium filter to the image using a 15 pixel x 15 pixel size
    % square to remove texture of the green background
    % (i.e. to remove faded/scraped parts of the assembly line belt)
    gsSmooth = medfilt2(gs, [15 15]);
    
    % Apply adaptive thresholding to the previously medium-filtered image
    % This calculates a threshold based on the local mean intensity of the
    % vicinity of each pixel that is evaluated. 
    % This actively searches for "bright", shiny surfaces (like the tubes)
    % against the background. 
    % The resulting threshold is then used with imbinarize() to create a binary
    % image
    T = adaptthresh(gsSmooth, 0.45, 'ForegroundPolarity', 'bright');
    shapeMask = imbinarize(gsSmooth, T);
    
    % Fill in "hole" regions within the tubes to make them appear solid on the
    % mask using imfill()
    % Remove small objects from the mask that are smaller than 500 pixels using
    % bwareaopen()
    shapeMask = imfill(shapeMask, 'holes');
    shapeMask = bwareaopen(shapeMask, 500);
    
    % Create a disk structural image with radius = 6 pixels
    % to create an open image with the mask. This serves to address cases where
    % tubes are falsely classified as one tube due to their close proximity.
    breakSE = strel('disk', 6);
    shapeMask = imopen(shapeMask, breakSE);
    
    % Sets the corners of the image to 0 to prevent noise from affecting the
    % image analysis
    cornerSize = 50;
    shapeMask(1:cornerSize, 1:cornerSize) = 0;
    shapeMask(1:cornerSize, end-cornerSize:end) = 0;
    shapeMask(end-cornerSize:end, 1:cornerSize) = 0;
    shapeMask(end-cornerSize:end, end-cornerSize:end) = 0;

    % Measure the shiny silver tubes based on the shape mask using
    % regionprops()
    % ***IMPORTANT***
    %****************
    % regionprops() objects serve as a structure that tracks several properties
    % of images. In this case we are tracking:
    % 1) Centroid --> To determine the center of each tube
    %
    % 2) MajorAxisLength --> To determine the length of each tube
    %
    % 3) MinorAxisLength --> To determine the width of each tube
    %
    % 4) Orientation --> To determine the angle at which each tube is oriented.
    %                    This is for tightly fitting a rectangle to each tube
    %                    in order to classify each object after processing is
    %                    complete. Otherwise, the rectangles would be misplaced
    %                    and not line up with each tube correctly.
    %
    % 5) Area --> To track how much Area each tube takes up in order to
    %             determine a mean Area, to which other objects in the image
    %             are compared against. Objects significantly smaller or larger than this
    %             mean area are classified as anomalies. 
    %
    % 6) Eccentricity --> To  determine how "elongated" an object is. This
    %                     address cases where the tube is cut off (i.e. similar width but much shorter), and hence
    %                     significantly shorter than a standard size tube.
    shapeStats = regionprops(shapeMask, 'Centroid', 'MajorAxisLength', 'MinorAxisLength', 'Orientation', 'Area', 'Eccentricity', 'Perimeter');
    
    % Number each entry
    for k = 1:length(shapeStats)
        shapeStats(k).Num = k;
    end

    % Assign significance to be false by default
    for k = 1:length(shapeStats)
        shapeStats(k).significant = false;
    end

    % Difference between rectangular perimeter and calculated perimeter,
    % Used to detect abnormalities such as dents which significantly
    % decreases this value
    for k = 1:length(shapeStats)
        shapeStats(k).PerimeterDifference = (2*shapeStats(k).MajorAxisLength + 2*shapeStats(k).MinorAxisLength) - shapeStats(k).Perimeter;
    end
    
    % =========================================================================
    % METRICS TRACKING & VISUALIZATION
    % =========================================================================
    hiddenFig = figure("Visible","off");
    imshow(img);
    hold on;
    
    % Initialize counters for the silver tubes
    numShapeAnomalies = 0;
    numNormalTubes = 0;
    
    % Calibration variables for average tube area
    % You must adjust these three numbers based on a "Good" image!
    minNormalArea = 2000;         % Anything smaller than this is ignored as noise
    maxNormalArea = 5200;       % Anything larger than this is flagged as crushed
    minNormalEccentricity = 0.950; % Anything less elongated than this is flagged
    maxNormalEccentricity = 0.999; % Anything more elongated than this is flagged
    minPerimeterDifference = 35; % Anything less than this is flagged (used to detect dents)
    minDoublePerimeterDifference = 100;
    maxTubeLen = 200;
    minTubeLen = 110;
    
    % Draw and Log Color Anomalies onto original image
    for c = 1:numColorAnomalies
        % The below statements will plot red boxes over the tubes flagged as a
        % color anomaly.
        box = colorStats(c).BoundingBox;
        plot([box(1), box(1)+box(3), box(1)+box(3), box(1), box(1)], ...
             [box(2), box(2), box(2)+box(4), box(2)+box(4), box(2)], 'r-', 'LineWidth', 2);
    end

    % Draw and Log Shape Anomalies vs. Normal Tubes
    for k = 1:length(shapeStats)
        
        thisArea = shapeStats(k).Area;
        
        % Apply a noise filter using "historical" tube statistics that are set
        % as constants in this script (adjust as deemed necessary to fine tune
        % this part of the code)
        if thisArea < minNormalArea
            continue; 
        end

        % labels this ROI as significant (used for data analysis)
        shapeStats(k).significant = true;
        
        thisEcc = shapeStats(k).Eccentricity;
        
        % Code to tightly fit a rectangle around each tube.
        center = shapeStats(k).Centroid;
        majorLen = shapeStats(k).MajorAxisLength;
        minorLen = shapeStats(k).MinorAxisLength;
        angle = -shapeStats(k).Orientation; 
        perimeterDifference = shapeStats(k).PerimeterDifference;

        L = majorLen / 2; W = minorLen / 2;
        x = [-L,  L,  L, -L, -L]; y = [-W, -W,  W,  W, -W];
        
        theta = deg2rad(angle);
        R = [cos(theta), -sin(theta); sin(theta),  cos(theta)];
        rotCoords = R * [x; y];
        
        boxX = rotCoords(1, :) + center(1);
        boxY = rotCoords(2, :) + center(2);

        
        % Classify and count the number of normal and abnormal tubes
        if ((((thisArea <= maxNormalArea) && (thisArea >= minNormalArea))... %Checks if the signular tube matches length requirements
            && ((thisEcc >= minNormalEccentricity) && (thisEcc <= maxNormalEccentricity))... % Checks to see if the tube aligns with Eccentricity min & max
            && (perimeterDifference >= minPerimeterDifference)...%Checks if Perimeter difference is large enough for a singular tube
            && (majorLen < maxTubeLen) && (majorLen > minTubeLen))... % Checks if tube meets length requirements to be classified as a singular tube
            || (majorLen >= maxTubeLen) && (perimeterDifference >= minDoublePerimeterDifference)); % Detects if a tube is "merged" with another and judges only its combined PD


            plot(boxX, boxY, 'g-', 'LineWidth', 2); % Flag as Normal
            
            numNormalTubes = numNormalTubes + 1;
        else

            plot(boxX, boxY, 'y-', 'LineWidth', 2); % Flag as Anomaly
            
            numShapeAnomalies = numShapeAnomalies + 1;
        end
    
        % Label each with its corresponding number for easier data lookup
        % from the output table
        
        text(center(1), center(2), string(shapeStats(k).Num), "FontSize", 14, "Color", 'black');
        
    end

    % Turn off hold after all bounding boxes are drawn 
    % onto the original image
    hold off;

    % Take a screenshot of the current plot/axis w/ gca (i.e. bounding boxes atop the
    % original image) to save the annotated picture for use in the "fancier" program output
    frame = getframe(gca);
    annotatedImage = frame.cdata;
    close(hiddenFig);

    % Print some results to the Command Window for troubleshooting purposes
    totalTubesInImage = numNormalTubes + numShapeAnomalies + numColorAnomalies;
     
end