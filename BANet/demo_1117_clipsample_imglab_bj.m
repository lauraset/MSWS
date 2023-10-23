%% �ü�Ӱ��ͱ�ǩ�����ص��ü�
% 2022.11.22: clip certall
clc;clear;
ipath = 'D:\change\beijing\';
fcode = 'bj';
respath = 'D:\change\beijing\basample\';
% ipath = 'D:\change\shanghai\';
% fcode = 'sh';
% respath = 'D:\change\data_sh\';

% 1. ��ȡmask
maskp = fullfile(ipath, 'mask'); % shape 
mask = uint8(nfreadenvi(maskp));% imshow(mask);
mask = mask'; % ת��
% ���أ�2022.12.13
iname = 'certpix';
fprintf('process: %s\n', iname);
certp = fullfile(ipath,'res_ba','diffpixcert.tif');
% ����2022.12.13
% iname = 'certobj';
% fprintf('process: %s\n', iname);
% certp = fullfile(ipath,'res_ba','diffobjcert.tif');
% ����+����
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

% 2. �ü�Ӱ��
N = 256;
stride = N/2; % ���ص�
% ��ȷ��ͼ
k=islice_mask_png(cert, mask, N ,stride, [respath, iname, '\','lab','_',fcode,'_']);
certc =cert;
certc(cert==1)=200; % negative
certc(cert==2)=255; % positive
certc(cert==3)=128; % ignore
k=islice_mask_png(certc, mask, N ,stride, [respath, iname,'c\','lab','_',fcode,'_']);
toc;