clear all;close all;clc;
fileFolder=fullfile('segmim');
dirOutput=dir(fullfile(fileFolder,'*.jpg'));

for i = 1 : 8189
    local_segmim = strcat('segmim/'  ,dirOutput(i).name);
    local_xfc_segmim = strcat('XFC_segmim/' ,dirOutput(i).name);
    pict=imread(local_segmim);
    size_pict = size(pict);
    %pict_xfc = zeros(size_pict(2),size_pict(2),3);
    pict_xfc1 = cov( double(pict(:,:,1)));
    pict_xfc2 = cov( double(pict(:,:,2)));
    pict_xfc3 = cov( double(pict(:,:,3)));       
    pict_xfc = cat(3,pict_xfc1,pict_xfc2,pict_xfc3);
    c_output = mat2gray(pict_xfc);
    
    imwrite(c_output,local_xfc_segmim);
end

