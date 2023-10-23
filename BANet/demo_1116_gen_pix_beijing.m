%% 生成变化的区域
% 1.预测结果直接求差
%以北京为例
ipath = 'D:\change\beijing\pred_ba\';
respath = 'D:\change\beijing\res_ba\';
if ~isfolder(respath)
    mkdir(respath);
end
t1name = 'rrm_tlcmulti4_adeleimg18_seg.tif';
t2name = 'rrm_tlcmulti4_adeleimg28_seg.tif';

p1 = [ipath, t1name];
[t1, R] = geotiffread(p1);
info = geotiffinfo(p1);

% t1 = imread([ipath, t1name]);
t2 = imread([ipath, t2name]);

t1= single(t1)/255.0;
t2 = single(t2)/255.0;
diff = (t2-t1); % -1,0,1
diff_ = zeros(size(t1));
diff_(diff==-1)=1; % demolished, negative
diff_(diff==1)=2; % newly-built, positive
% diff_(diff==0 & t1==1)=3; % unchange

resname = [respath,'diffpix.tif'];
geotiffwrite(resname, uint8(diff_), R, ...
    'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag);
% 面积滤波:分别对 新增和拆除的滤波
area_thr = 500; % 对于建筑区比较有效，分别对 正、负变化区域进行 面积滤波
res1 = area_filter(diff_==1, area_thr);
res2 = area_filter(diff_==2, area_thr);
res = res1+ 2* res2;
% save
resname= fullfile(respath,'diffpixarea.tif');
geotiffwrite(resname, uint8(res), R, ...
    'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag);

clear;
%% 计算像素差值的概率，得到不确定区域
addpath('ecognition');
resroot = 'D:\change\beijing\res_ba\';

iroot = 'D:\change\beijing\pred_ba\';
p1 = [iroot, 'rrm_tlcmulti4_adeleimg18.tif'];
p2 = [iroot, 'rrm_tlcmulti4_adeleimg28.tif'];

[f1,R]  = geotiffread(p1);
info= geotiffinfo(p1);
f2 = imread(p2);
f1 = single(f1)/255.0;
f2 = single(f2)/255.0;

res = f2-f1;
imshow(res,[]);

%% 自动阈值
t = multithresh(res(:),6);
res_seg = zeros(size(res),'uint8');
res_seg(res<=t(1))=1; % negative, 拆除的
res_seg(res>=t(end))=2; % positive, 新建的
% 面积阈值
area_thr = 1000; % 对于建筑区比较有效，分别对 正、负变化区域进行 面积滤波
res1 = area_filter(res_seg==1, area_thr);
res2 = area_filter(res_seg==2, area_thr);
res_segarea = res1+ 2* res2;

% 不确定的区域
res_uncert = zeros(size(res),'uint8'); % 经过面积滤除的区域，1,2均为确定的区域，0包含了背景和不确定区域
res_uncert(res>t(1) & res<t(2)) = 1; % 
res_uncert(res>t(end-1) & res<t(end)) = 1;
res_uncert = area_filter(res_uncert, area_thr);
res_uncert = res_segarea + 3* res_uncert;

imshow(res_uncert,[]);
resname= fullfile(resroot,'diffpixcert.tif');
geotiffwrite(resname, uint8(res_uncert), R, ...
    'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag);

