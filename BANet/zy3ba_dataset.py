import os
import os.path as osp
from os.path import join
import pandas as pd
from glob import glob

def get_file(ipath, respath):
    imgpath = join(ipath,'img')

    img_pos = os.listdir(join(ipath, 'pos'))
    img_pos.sort()
    img_neg = os.listdir(join(ipath, 'neg'))
    img_neg.sort()
    imglist_pos = [join(imgpath, i[:-4]+'.tif') for i in img_pos]
    imglist_neg = [join(imgpath, i[:-4]+'.tif') for i in img_neg]
    imglist = imglist_pos + imglist_neg
    lablist = [0 for i in range(len(img_pos))] + [1 for i in range(len(img_neg))]

    df = pd.DataFrame({'img':imglist, 'lab': lablist})
    df.to_csv(respath, sep=',', header=None, index=False)

# 2023.6.6: exlude some cities
def get_file_exlude(ipath, respath, fcode=['bj', 'sh']):
    imgpath = join(ipath,'img')
    img_pos = os.listdir(join(ipath, 'pos'))
    img_pos.sort()
    img_neg = os.listdir(join(ipath, 'neg'))
    img_neg.sort()
    # exlude some cities
    imglist_pos = [join(imgpath, i[:-4]+'.tif') for i in img_pos if i[4:6] not in fcode]
    imglist_neg = [join(imgpath, i[:-4]+'.tif') for i in img_neg if i[4:6] not in fcode]
    imglist = imglist_pos + imglist_neg
    lablist = [0 for i in range(len(imglist_pos))] + [1 for i in range(len(imglist_neg))]
    # save
    df = pd.DataFrame({'img':imglist, 'lab': lablist})
    df.to_csv(respath, sep=',', header=None, index=False)


def get_file_512(ipath, respath):
    imgpath = join(ipath, 'img')

    img_all = os.listdir(imgpath)

    imglist = [join(imgpath, i[:-4]+'.tif') for i in img_all]

    lablist = [0 for i in range(len(imglist))]

    df = pd.DataFrame({'img':imglist, 'lab': lablist})
    df.to_csv(respath, sep=',', header=None, index=False)


def split_df(df, value=0, split_rate=0.9):
    df1 = df.loc[df[1]==value].sample(frac=1, random_state=1) # extract and shuffle
    num_train = int(len(df1)*split_rate)
    return df1[:num_train], df1[num_train:]


def split_data(datalist_path, split_rate=0.9, id='2', n1='train', n2='test'):
    data_dir = os.path.dirname(datalist_path)
    base_name = os.path.basename(datalist_path)[:-4]
    train_path = join(data_dir, base_name+'_'+ n1 + id +'.csv')
    test_path = join(data_dir, base_name+'_'+ n2 + id +'.csv')
    if os.path.exists(train_path) and os.path.exists(test_path):
        print('train and test list exist')
        return
    else:
        df = pd.read_csv(datalist_path, sep=',', header=None)
        # stratified random sampling
        df0_trn, df0_tst = split_df(df, 0, split_rate)
        df1_trn, df1_tst = split_df(df, 1, split_rate)
        df_train = pd.concat([df0_trn, df1_trn], axis=0)#.sample(frac=1, random_state=1) # concate and shuffle
        df_test = pd.concat([df0_tst, df1_tst], axis=0)#.sample(frac=1, )
        df_train.to_csv(train_path, index=False, sep=',', header=None)
        df_test.to_csv(test_path, index=False, sep=',', header=None)
        print('success')

def extract_neg(list1,respath):
    df1 = pd.read_csv(list1, header=None, delimiter=',')
    df2 = df1[df1[1]==1]
    df2.to_csv(respath, header=None, index=False)

def extract_pos(list1,respath):
    df1 = pd.read_csv(list1, header=None, delimiter=',')
    df2 = df1[df1[1]==0]
    df2.to_csv(respath, header=None, index=False)


def get_img_lab_test(datapath, listname):
    datalist = join(datapath, listname)
    if os.path.exists(datalist):
        print('datalist exists')
        return
    else:
        imgpath = join(datapath, "img") # img
        labpath = join(datapath, "lab")
        assert os.path.isdir(imgpath) == True
        imglist = glob(join(imgpath, "*.tif"))
        with open(datalist,"w") as f: # append
            for img in imglist:
                ibase = os.path.basename(img)
                mask = os.path.join(labpath, ibase[:-4]+'.png')
                f.write(img+","+ mask+"\n")

if __name__=="__main__":
    # 2023.6.6: major revision
    # exlude four cities from training test
    ipath = r'E:\yinxcao\ZY3LC\datanew8bit'
    respath = join(ipath, 'datalist_posneg.csv')
    # exclude 4 cities: beijing, shanghai, xi'an, kunming
    fcode = ['bj', 's4', 's7', 'sh', 'xa', 'km']
    # get_file_exlude(ipath, respath, fcode)

    # train/val=0.6:0.4
    # split_data(respath, split_rate=0.6, id='_0.6')

    # choose 0.1 from test for validation
    # respath = join(ipath, 'datalist_posneg_test_0.6.csv')
    split_data(respath, split_rate=0.25, id='_0.25', n1= 'val', n2='test')

    list1 =join(ipath, 'datalist_posneg_train_0.6.csv')
    extract_pos(list1, list1[:-4]+'_pos.csv')
    extract_neg(list1, list1[:-4]+'_neg.csv')


    # ipath = r'E:\yinxcao\ZY3LC\datanew8bit'
    # respath = join(ipath, 'datalist_posneg.csv')
    # get_file(ipath, respath)

    # only use 50% data for experiment
    # split_data(respath, split_rate=0.5, id='_0.5')

    # train/val=0.6:0.4
    # print(len(df))
    # respath = join(ipath, 'datalist_posneg_train_0.5.csv')
    # split_data(respath, split_rate=0.6, id='_0.6')

    # choose 0.1 from test for validation
    # respath = join(ipath, 'datalist_posneg_test_0.6.csv')
    # split_data(respath, split_rate=0.25, id='_0.25', n1= 'val', n2='test')

    # 2022.9.22
    # list1 =join(ipath, 'datalist_posneg_train_0.6.csv')
    # extract_pos(list1, list1[:-4]+'_pos.csv')
    # extract_neg(list1, list1[:-4]+'_neg.csv')

    # 2022.9.29
    # ipath = r'E:\yinxcao\ZY3LC\datanew8bit_512'
    # respath = join(ipath, 'datalist_all.csv')
    # get_file_512(ipath, respath)

    # 2022.10.22
    # datapath = r'E:\yinxcao\ZY3LC\datanew8bit\test_forseg'
    # listname = join(datapath, 'testlist.csv')
    # get_img_lab_test(datapath, listname)







