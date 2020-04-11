close all;
clear all;

%% set parameters
up_scale = 8;
% dataset = 'Set14/';
% testfolder = ['Test/' dataset];
% resultfolder=['Result/EDSR_x' int2str(up_scale) '/' dataset];
testfolder = ['../Dataset/Set5/']
resultfolder=['../output/2020-01-09_01:00:17.430236/180000/']
%resultfolder=['Input/' int2str(up_scale) 'x/' dataset];

%% flat or deconv
filepaths = dir(fullfile(testfolder,'*.png'));
psnr_drbpsr = zeros(length(filepaths),1);
ssim_drbpsr = zeros(length(filepaths),1);
fsim_drbpsr = zeros(length(filepaths),1);

parfor i = 1 : length(filepaths)
   [add,imname,type] = fileparts(filepaths(i).name);
    im = imread([testfolder imname type]);
    
    
    %% remove border
    im_drl = shave(imread([resultfolder imname '.png']), [up_scale, up_scale]);
    %im_drl = imresize(imresize(imread([testfolder imname type]),1/up_scale,'bicubic'),up_scale,'bicubic');
    im_gnd = shave(imread([testfolder imname type]), [up_scale, up_scale]);
    if size(im_gnd,3) == 1
        im_gnd=im_gnd(:,:,[1 1 1]);
    end

    %% compute PSNR
    imname
    size(im_gnd)
    size(im_drl)
    psnr_drbpsr(i) = compute_psnr(im_gnd,im_drl);
    
    %% compute FSIM
    fsim_drbpsr(i) = FeatureSIM(im_gnd,im_drl);
     
     %% compute SSIM
    ssim_drbpsr(i) = ssim_index(im_gnd,im_drl);

end

fprintf('DRBPSR: %f , %f , %f \n', mean(psnr_drbpsr), mean(ssim_drbpsr), mean(fsim_drbpsr));