%% ���ɲ�ȷ����֪��α��ǩ
% ȷ���������ԣ�����Ľ������ȷ���������ԣ����صĽ��
% resroot = 'D:\change\beijing\res_ba\';
% resroot = 'D:\change\shanghai\res_ba\';
% resroot = 'D:\change\kunming\res_ba\';
resroot = 'D:\change\xian\res_ba\';
obj = 'diffobjarea.tif';
pix = 'diffpixarea.tif';
p1 = [resroot, obj];
p2 = [resroot, pix];

[f1,R]  = geotiffread(p1);
info= geotiffinfo(p1);
f2 = imread(p2);

% �ϳ�uncert-aware labels
res = f1;
res(f1==0 & f2>0)=3; % ���ɿ����򣺶����޶������У��������龯����©
res(f1>0 & f2==0)=3; % ���ɿ����򣺶����ж������ޣ���������©����©

resname= fullfile(resroot,'diffobjpixcert.tif');
geotiffwrite(resname, uint8(res), R, ...
    'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag);
