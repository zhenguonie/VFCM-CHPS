function [ms, ld] = ssim_ld_eval(A1, ref)

tarea=598400;

% imread and rgb2gray
A1 = rgb2gray(A1);
ref = rgb2gray(ref);

% resize
b = sqrt(tarea/size(ref,1)/size(ref,2));
ref = imresize(ref,b);
A1 = imresize(A1,[size(ref,1),size(ref,2)]);

% calculate
[ms,ld] = evalUnwarp(A1,ref);
