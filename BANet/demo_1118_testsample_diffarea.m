%% �ü��ο��ı仯����2022118
% ÿ������10��1024x1024������
clc;clear;
% icity = 'beijing';
% fcode = 'bj';
% icity = 'shanghai';
% fcode = 'sh';
icity = 'kunming';
fcode = 'km';
% icity = 'xian';
% fcode = 'xa';

ipath = ['D:\change\',icity,'\sample_ba\'];

region = imread([ipath, 'region10valid.tif']);
region(region==255)=0;
imshow(region,[]);

N = 256;
stride = N;
respath = ['D:\change\',icity, '\basample\testdata\'];
if ~isfolder(respath)
    mkdir(respath);
end
%% pixel label
% iname = 'diffpixarea'; % ���ؼ�
% iname = 'diffobjarea'; % ����
% iname = 'diffobjpixcert'; % ����
% iname = 'cd'; % ����
iname = 'diffpix'; % ���ڷ����Ƚ�

mkdir([respath, iname]);
mkdir([respath, iname,'c']);

diffpix = imread(['D:\change\',icity,'\res_ba\',iname,'.tif']);
% diffpix = 2*diffpix;
k=islice_test_png(diffpix, region, N ,stride, [respath, iname,'\lab_',fcode,'_']);
% color;
diffpixc =diffpix;
diffpixc(diffpix==1)=200; % 128
diffpixc(diffpix==2)=255;
diffpixc(diffpix==3)=128; % uncert
k=islice_test_png(diffpixc, region, N ,stride, [respath, iname,'c\lab_',fcode,'_']);
