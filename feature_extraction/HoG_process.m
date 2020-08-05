clear all;close all;clc;
fileFolder=fullfile('segmim');
dirOutput=dir(fullfile(fileFolder,'*.jpg'));

for i = 1 : 8189
    local_segmim = strcat('segmim/'  ,dirOutput(i).name);
    local_hog_segmim = strcat('hog_segmim/' ,dirOutput(i).name);

	pict=imread(local_segmim);
    
    [featureVector,hogVisualization] = extractHOGFeatures(pict);  
    %figure;  
    %imshow(pict);  
    % hold on;  
    figure; 
    plot(hogVisualization);  
    saveas(gcf,local_hog_segmim);
    close all;
    %imwrite(hogVisualization,local_hog_segmim);
end



