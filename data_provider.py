#!/usr/local/bin/python3

# -*-coding:utf-8 -*-


import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
import datetime as dt

import config
from config import station_info
from config import factor_file_name_dict
from config import pm25_file_name
from config import days_info
from config import args

class DataProvider(object):
    def __init__(self):
        # self.factor_dict = None
        # self.pm_df = None
        self.data_input = None #[step,station,factors+target]
        self.data_target = None
        self.data_pm = None
        self.init_step = 0
        self.step = self.init_step
        self.train_num = 700
        self.test_num = 29


    def load_data(self):
        # load factor file
        factor_df_dict = {}
        for factor, factor_file_name in factor_file_name_dict.items():
            temp_path = args.path_prefix_data + '/' + factor_file_name
            temp_df = pd.read_excel(temp_path)
            # use forward value to fill nan to fill the first value use bfill
            temp_df = temp_df.fillna(method='ffill').fillna(method='bfill')
            print('load factor file %s successfully.. shape = %s' %(factor,str(temp_df.shape)))
            # ensure the order of attribute
            factor_df_dict[factor] = temp_df[station_info]

        # self.factor_dict = factor_df_dict

        # load pm25 file
        path_pm = args.path_prefix_data + '/' + pm25_file_name
        pm_df = pd.read_excel(path_pm)
        pm_df = pm_df.fillna(method='ffill').fillna(method='bfill')

        #handle the datetime field
        def get_mon_day(delt_days):
            date_start = dt.datetime.strptime('2014-1-1','%Y-%m-%d')
            date_delt = dt.timedelta(days = delt_days)
            res = date_start + date_delt
            return res.month,res.day

        def get_matrix_date(serise_field):
            ss = serise_field.values.reshape(-1,1)
            temp_data = np.zeros((ss.shape[0],args.station_num))
            temp_data = temp_data + ss
            return temp_data

        mon_station = get_matrix_date(pm_df[days_info].apply(lambda x :get_mon_day(x)[0]))
        day_station = get_matrix_date(pm_df[days_info].apply(lambda x: get_mon_day(x)[1]))
        
        # select and ensure the order of the attribute
        pm_df = pm_df[station_info]
        
       

        print('load pm file done.. shape = %s' % str(pm_df.shape))
        # self.pm_df = pm_df
        # print(pm_df.columns)

        # concat the factor and pm to get the input format
        data_input = pm_df.values
        for factor, temp_df in factor_df_dict.items():
            data_input = np.concatenate((data_input,temp_df.values),axis = 1)
        #concat the mon day data to get the input format
        for temp_matrix in [mon_station, day_station]:
            data_input = np.concatenate((data_input,temp_matrix),axis = 1)

        print('conat done shape is %s ' % str(data_input.shape))

        # TODO: delete many nan step
        # # delete many nan step
        # nan_num_each_step = np.sum(np.isnan(data_input).astype(np.int32),axis = 1)
        # print(nan_num_each_step)

        data_input = data_input.reshape((-1,args.feature_num,args.station_num))
        print('reshape data_input shape to %s which represents [step,factor,station]..' % str(data_input.shape))
        data_input = data_input.swapaxes(1,2)
        print('swap the factor axis and station axis now input_data shape is %s which represents '
              '[step,station,factors]..' % str(data_input.shape))

        # standerizaion by std_dev
        data_input_std = np.zeros(data_input.shape)
        for index_station in range(args.station_num):
            for index_factor in range(args.feature_num):
                temp_data = data_input[:,index_station,index_factor]
                temp_mean = np.mean(temp_data)
                temp_std = np.std(temp_data)
                # temp_mean = 0
                # temp_std = 1
                data_input_std[:,index_station,index_factor] = (temp_data - temp_mean) / temp_std


        # target feature not standardrization
        self.data_pm = pm_df.values
        # data_target_std = self.data_pm - np.mean(self.data_pm, axis=0)
        data_target_std = self.data_pm
        self.data_input = data_input_std[:-1,:,:]
        self.data_target = data_target_std[1:,:]

        print('load data done! input shape is %s, target shape is %s' % \
              (str(self.data_input.shape),str(self.data_target.shape)))


    # def diff_restore(diff_val, _start, _end):
    #     data_ori = self.data_pm[_start:_end,:]
    #     assert data_ori.shape == diff_val.shape
    #     pre_val = diff_val + data_ori
    #     mse = np.sum((pre_val - data_ori) ** 2) / pre_val.size
    #     return pre_val,mse
        



    def next_batch(self,_start = None , _step_num = None):
        if _step_num:
            step_num = _step_num
        else:
            step_num = args.step_num
        if _start:
            start = _start
        else:
            start = self.step
        end = start + step_num
        if end > self.train_num:
            self.step = self.init_step
            start = self.step
            end = start + step_num
        input = self.data_input[start:end,:,:].reshape((-1,args.feature_num*args.station_num))
        target = self.data_target[start:end,:]
        self.step += 1
        return input,target

    def get_test(self):
        start = self.train_num
        # end = self.data_target.shape[0]
        end = start + self.test_num
        input = self.data_input[start:end, :, :].reshape((-1, args.feature_num * args.station_num))
        target = self.data_target[start:end, :]
        return input, target

if __name__ == '__main__':
    dp = DataProvider()
    dp.load_data()
    input,target = dp.next_batch()
    print(input.shape,target.shape)
    print(input)













