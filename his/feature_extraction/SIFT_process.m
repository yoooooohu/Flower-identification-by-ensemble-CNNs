% clear all;close all;clc;
fileFolder=fullfile('segmim');
dirOutput=dir(fullfile(fileFolder,'*.jpg'));

local_segmim1 = strcat('segmim/'  ,dirOutput(1).name);
local_segmim2 = strcat('segmim/'  ,dirOutput(2).name);
size_pict=zeros(8189,2);
 for i = 1 : 2%8189
     local_segmim = strcat('segmim/'  ,dirOutput(i).name);
     local_hog_segmim = strcat('sift_segmim/' ,dirOutput(i).name);
     local_hog_segmim(26:28) = 'mat';
     pict=imread(local_segmim);
     try
        [des1,loc1] = getFeatures(pict);
        size_pict(i,:) = size(des1);
        drawFeatures(pict,loc1);
        
        close all;
     catch
        continue;
     end
     reshape_sift = des1;%(1:700,:);
     save(local_hog_segmim, 'reshape_sift');     
 end


% clear
% tic
% img1 = imread('scene.pgm');
% img2 = imread('book.pgm');
% [des1,loc1] = getFeatures(img1);
% [des2,loc2] = getFeatures(img2);
% matched = match(des1,des2);
% drawFeatures(img1,loc1);
% drawFeatures(img2,loc2);
% drawMatched(matched,img1,img2,loc1,loc2);
% toc
