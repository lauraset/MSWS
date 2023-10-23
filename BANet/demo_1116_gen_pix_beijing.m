%% ���ɱ仯������
% 1.Ԥ����ֱ�����
%�Ա���Ϊ��
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
% ����˲�:�ֱ�� �����Ͳ�����˲�
area_thr = 500; % ���ڽ������Ƚ���Ч���ֱ�� �������仯������� ����˲�
res1 = area_filter(diff_==1, area_thr);
res2 = area_filter(diff_==2, area_thr);
res = res1+ 2* res2;
% save
resname= fullfile(respath,'diffpixarea.tif');
geotiffwrite(resname, uint8(res), R, ...
    'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag);

clear;
%% �������ز�ֵ�ĸ��ʣ��õ���ȷ������
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

%% �Զ���ֵ
t = multithresh(res(:),6);
res_seg = zeros(size(res),'uint8');
res_seg(res<=t(1))=1; % negative, �����
res_seg(res>=t(end))=2; % positive, �½���
% �����ֵ
area_thr = 1000; % ���ڽ������Ƚ���Ч���ֱ�� �������仯������� ����˲�
res1 = area_filter(res_seg==1, area_thr);
res2 = area_filter(res_seg==2, area_thr);
res_segarea = res1+ 2* res2;

% ��ȷ��������
res_uncert = zeros(size(res),'uint8'); % ��������˳�������1,2��Ϊȷ��������0�����˱����Ͳ�ȷ������
res_uncert(res>t(1) & res<t(2)) = 1; % 
res_uncert(res>t(end-1) & res<t(end)) = 1;
res_uncert = area_filter(res_uncert, area_thr);
res_uncert = res_segarea + 3* res_uncert;

imshow(res_uncert,[]);
resname= fullfile(resroot,'diffpixcert.tif');
geotiffwrite(resname, uint8(res_uncert), R, ...
    'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag);

