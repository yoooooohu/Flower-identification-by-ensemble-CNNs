% clear all;close all;clc;
fileFolder=fullfile('sift_segmim');
dirOutput=dir(fullfile(fileFolder,'*.mat'));

local_segmim1 = strcat('sift_segmim/'  ,dirOutput(1).name);
local_segmim2 = strcat('sift_segmim/'  ,dirOutput(2).name);
size_pic=zeros(7888,2);
 for i = 1 : 7888
     local_segmim = strcat('sift_segmim/'  ,dirOutput(i).name);
     local_hog_segmim = strcat('sift_segmim2/' ,dirOutput(i).name);
     local_hog_segmim(27:29) = 'jpg';
     load(local_segmim); 
     size_pict=size(reshape_sift);
     if size_pict(1) < 128
        reshape_sift = repmat( reshape_sift , ceil(128/size_pict(1)) , 1 );
     end
     reshape_sift = reshape_sift(1:128,:);
     size_pic(i,:) = size(reshape_sift);
     imwrite(mat2gray(reshape_sift), local_hog_segmim);
     %save(local_hog_segmim, 'reshape_sift');     
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
