%% 面向对象投影预测的概率
addpath('ecognition');
segpath = 'D:\change\beijing\ecognition\seg20181006_50.tif';
% segpath = 'D:\change\beijing\ecognition\segt12_50.tif'; 同时叠加t1和t2
resroot = 'D:\change\beijing\res_ba\';

obj = seg2label(segpath);
obj = obj';
stats = regionprops(obj, 'PixelIdxList');

iroot = 'D:\change\beijing\pred_ba\';
p1 = [iroot, 'rrm_tlcmulti4_adeleimg18.tif'];
p2 = [iroot, 'rrm_tlcmulti4_adeleimg28.tif'];

[f1,R]  = geotiffread(p1);
info= geotiffinfo(p1);
f2 = imread(p2);
f1 = single(f1)/255.0;
f2 = single(f2)/255.0;

%% 计算两个特征的差值
num = length(stats);
res=zeros(size(f1),'single');
for i=1:num
    id=stats(i).PixelIdxList;
    tmp1=f1(id);
    tmp2=f2(id);
    res(id)=mean(tmp2,1)- mean(tmp1,1);%each column
end
imshow(res);
resname= fullfile(resroot,'diffobjprob.tif');
geotiffwrite(resname, uint8(abs(res)*255), R, ...
    'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag);

%% 自动阈值
t = multithresh(res(:),6);
res_seg = zeros(size(res),'uint8');
res_seg(res<=t(1))=1; % negative, 拆除的
res_seg(res>=t(end))=2; % positive, 新建的
imshow(res_seg,[]);
resname= fullfile(resroot,'diffobj.tif');
geotiffwrite(resname, uint8(res_seg), R, ...
    'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag);

% 面积滤波
area_thr = 1000; % 对于建筑区比较有效，分别对 正、负变化区域进行 面积滤波
res1 = area_filter(res_seg==1, area_thr);
res2 = area_filter(res_seg==2, area_thr);
res_segarea = res1+ 2* res2;
% save
imshow(res_segarea,[]);
resname= fullfile(resroot,'diffobjarea.tif');
geotiffwrite(resname, uint8(res_segarea), R, ...
    'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag);

% 不确定的区域赋值为 3
res_uncert = res_segarea; % 经过面积滤除的区域，1,2均为确定的区域，0包含了背景和不确定区域
res_uncert(res>t(1) & res<t(2)) = 3; % 
res_uncert(res>t(end-1) & res<t(end)) = 3;
resname= fullfile(resroot,'diffobjcert.tif');
geotiffwrite(resname, uint8(res_uncert), R, ...
    'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag);
