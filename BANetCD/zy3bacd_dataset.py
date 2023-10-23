import os
from os.path import join
from glob import glob
import pandas as pd

def get_imglist(ipath, city='bj'):
    '''
    :param ipath:
    :return:
    '''
    imglist = glob(join(ipath, 'img1', '*'+city+'*.tif'))
    imglist.sort()
    df= pd.DataFrame({'imglist': imglist})
    return df, imglist



if __name__=="__main__":
    # beijing
    '''
    # all images
    ipath = r'E:\yinxcao\weaksup\BAdata'
    df, imglist = get_imglist(ipath, 'bj')
    # df.to_csv(join(ipath, 'imglist.csv'), header=None, index=False)
    # test images
    ipath_test = r'E:\yinxcao\weaksup\BAdata\testdata'
    df_test, imglist_test = get_imglist(ipath_test, 'bj')
    df_test.to_csv(join(ipath_test, 'imglist_test.csv'), header=None, index=False)
    # train images
    imglist_train = []
    for i in imglist:
        iname = os.path.basename(i)
        tmp = os.path.join(ipath_test, 'img1', iname)
        if not tmp in imglist_test:
            imglist_train.append(i)
    df_train = pd.DataFrame({'imglist': imglist_train})
    df_train.to_csv(join(ipath, 'imglist_train.csv'), header=None, index=False)
    '''
    # shanghai
    # all images
    # ipath = r'E:\yinxcao\weaksup\BAdata_sh'
    # icity = 'sh'
    # 2023.6.16: kunming
    ipath = r'E:\yinxcao\weaksup\BAdata_km'
    icity = 'km'
    # 2023.6.18: xian
    # 2023.6.18: kunming
    ipath = r'E:\yinxcao\weaksup\BAdata_xa'
    icity = 'xa'

    df, imglist = get_imglist(ipath, icity)
    df.to_csv(join(ipath, 'imglist.csv'), header=None, index=False)
    # test images
    ipath_test = os.path.join(ipath, 'testdata')
    df_test, imglist_test = get_imglist(ipath_test, icity)
    df_test.to_csv(join(ipath, 'imglist_test.csv'), header=None, index=False)
    # train images
    imglist_train = []
    for i in imglist:
        iname = os.path.basename(i)
        tmp = os.path.join(ipath_test, 'img1', iname)
        if not tmp in imglist_test:
            imglist_train.append(i)
    df_train = pd.DataFrame({'imglist': imglist_train})
    df_train.to_csv(join(ipath, 'imglist_train.csv'), header=None, index=False)