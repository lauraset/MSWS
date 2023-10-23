%% 裁剪影像和标签：半重叠裁剪
% 2022.11.22: clip certall
clc;clear;
ipath = 'D:\change\beijing\';
fcode = 'bj';
respath = 'D:\change\beijing\basample\';
% ipath = 'D:\change\shanghai\';
% fcode = 'sh';
% respath = 'D:\change\data_sh\';

% 1. 读取mask
maskp = fullfile(ipath, 'mask'); % shape 
mask = uint8(nfreadenvi(maskp));% imshow(mask);
mask = mask'; % 转置
% 像素，2022.12.13
iname = 'certpix';
fprintf('process: %s\n', iname);
certp = fullfile(ipath,'res_ba','diffpixcert.tif');
% 对象，2022.12.13
% iname = 'certobj';
% fprintf('process: %s\n', iname);
% certp = fullfile(ipath,'res_ba','diffobjcert.tif');
% 对象+像素
% iname = 'certobjpix';
% fprintf('process: %s\n', iname);
% certp = fullfile(ipath,'res_ba','diffobjpixcert.tif');
% iname = 'certall';
% fprintf('process: %s\n', iname);
% certp = fullfile(ipath,'res_ba','certall.tif');

% iname = 't1seg';
% fprintf('process: %s\n', iname);
% certp = fullfile(ipath,'res_ba',[iname,'.tif']);

% iname = 't2seg';
% fprintf('process: %s\n', iname);
% certp = fullfile(ipath,'res_ba',[iname,'.tif']);

% iname = 'cd';
% fprintf('process: %s\n', iname);
% certp = fullfile(ipath,'res_ba',[iname,'.tif']);

if ~isfolder([respath, iname])
    mkdir([respath, iname]);
end
if ~isfolder([respath, iname,'c'])
    mkdir([respath, iname,'c']);
end
tic;
cert = imread(certp);

% 2. 裁剪影像
N = 256;
stride = N/2; % 半重叠
% 不确定图
k=islice_mask_png(cert, mask, N ,stride, [respath, iname, '\','lab','_',fcode,'_']);
certc =cert;
certc(cert==1)=200; % negative
certc(cert==2)=255; % positive
certc(cert==3)=128; % ignore
k=islice_mask_png(certc, mask, N ,stride, [respath, iname,'c\','lab','_',fcode,'_']);
toc;