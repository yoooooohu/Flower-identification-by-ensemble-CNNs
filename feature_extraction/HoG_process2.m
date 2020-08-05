clear all;close all;clc;
fileFolder=fullfile('segmim');
dirOutput=dir(fullfile(fileFolder,'*.jpg'));

for i = 1 : 8189
    local_segmim = strcat('segmim/'  ,dirOutput(i).name);
    local_hog_segmim = strcat('hog_segmim2/' ,dirOutput(i).name);
    local_hog_segmim(26:28) = 'mat';
	pict=imread(local_segmim);
    size_pict = size(pict);
    
    [featureVector,hogVisualization] = extractHOGFeatures(pict);  
    
    reshape_hog = reshape(featureVector,36,floor(size_pict(1)/8) - 1,floor(size_pict(2)/8) - 1);
    % A2 = squeeze(sum(reshape_hog,1));   %squeeze降维函数
    % imshow(mat2gray(A2));
    %reshape_hog = permute(reshape_hog,[3,2,1]);     %交换第一维第三维
    size_reshape_hog = size(reshape_hog);
    reshape_hog = reshape_hog(:, (floor(size_reshape_hog(2)/2) - 14) : (floor(size_reshape_hog(2)/2) + 14) , (floor(size_reshape_hog(3)/2) - 14) : (floor(size_reshape_hog(3)/2) + 14) );
    save(local_hog_segmim, 'reshape_hog');
    
end

