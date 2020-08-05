clear all;close all;clc;
fileFolder=fullfile('segmim');
dirOutput=dir(fullfile(fileFolder,'*.jpg'));

for i = 1 : 2   %8189
    local_segmim = strcat('segmim/'  ,dirOutput(i).name);
    local_hsv_segmim = strcat('hsv_segmim/' ,dirOutput(i).name);
    pict=imread(local_segmim);
    pict = rgb2hsv(pict);
    %imshow(mat2gray(pict))      %œ‘ æª“∂»Õº
    c_output = mat2gray(pict);
    
    
    imwrite(c_output,local_hsv_segmim);
end

