%% 生成不确定感知的伪标签
% 确定区域来自：对象的结果；不确定区域来自：像素的结果
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

% 合成uncert-aware labels
res = f1;
res(f1==0 & f2>0)=3; % 不可靠区域：对象无而像素有，可能是虚警、遗漏
res(f1>0 & f2==0)=3; % 不可靠区域：对象有而像素无，可能是遗漏、遗漏

resname= fullfile(resroot,'diffobjpixcert.tif');
geotiffwrite(resname, uint8(res), R, ...
    'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag);
