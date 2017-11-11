#! /usr/local/bin/python3
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from config import station_info
from config import args



path_factor_list = ['Irad_bj.xlsx','SHum_bj.xlsx','SRad_bj.xlsx','prec_bj.xlsx',
                    'pres_bj.xlsx','temp_bj.xlsx','wind_bj.xlsx']

path_mask = 'sta_msk.csv'

def load_factor_data(path_factor_list,path_mask):
    '''
    transform the origin data, extract the given station data
    :param path_factor_list:
    :param path_mask:
    :return:
    '''
    msk = np.loadtxt(args.path_prefix_data + '/' + path_mask,delimiter=',',dtype=np.int32)
    for temp_path in path_factor_list:
        temp_df = pd.read_excel(args.path_prefix_data + '/'+ temp_path, header=None)
        val = temp_df.values.T

        val_msk = val[msk>0].T
        trans_df = pd.DataFrame(data=val_msk,columns=station_info)
        trans_df.to_excel(args.path_prefix_data + '/trans' + temp_path)

if __name__ == '__main__':
    load_factor_data(path_factor_list,path_mask)

