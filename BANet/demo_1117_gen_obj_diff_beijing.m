%% �������ͶӰԤ��ĸ���
addpath('ecognition');
segpath = 'D:\change\beijing\ecognition\seg20181006_50.tif';
% segpath = 'D:\change\beijing\ecognition\segt12_50.tif'; ͬʱ����t1��t2
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

%% �������������Ĳ�ֵ
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

%% �Զ���ֵ
t = multithresh(res(:),6);
res_seg = zeros(size(res),'uint8');
res_seg(res<=t(1))=1; % negative, �����
res_seg(res>=t(end))=2; % positive, �½���
imshow(res_seg,[]);
resname= fullfile(resroot,'diffobj.tif');
geotiffwrite(resname, uint8(res_seg), R, ...
    'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag);

% ����˲�
area_thr = 1000; % ���ڽ������Ƚ���Ч���ֱ�� �������仯������� ����˲�
res1 = area_filter(res_seg==1, area_thr);
res2 = area_filter(res_seg==2, area_thr);
res_segarea = res1+ 2* res2;
% save
imshow(res_segarea,[]);
resname= fullfile(resroot,'diffobjarea.tif');
geotiffwrite(resname, uint8(res_segarea), R, ...
    'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag);

% ��ȷ��������ֵΪ 3
res_uncert = res_segarea; % ��������˳�������1,2��Ϊȷ��������0�����˱����Ͳ�ȷ������
res_uncert(res>t(1) & res<t(2)) = 3; % 
res_uncert(res>t(end-1) & res<t(end)) = 3;
resname= fullfile(resroot,'diffobjcert.tif');
geotiffwrite(resname, uint8(res_uncert), R, ...
    'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag);
