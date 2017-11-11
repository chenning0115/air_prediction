#! /usr/local/bin/python3


# import argparse
#
# parser = argparse.ArgumentParser()
#
# parser.add_argument('--path_prefix_data',default='../data',type=str ,help='the prefix of data path')
# parser.add_argument('--path_predix_summary', type=str, default='../summary')
#
# parser.add_argument('--feature_num',type = int, default= 8, help='the input feature num of the network'                                                               'should be the sum of factor_num and auto-relation_num')
# parser.add_argument('--station_num',type = int, default=21, help = 'the number of station')
# parser.add_argument('--lstm_num_units', type = int, default=64, help = 'the number of the lstm unit num')
# parser.add_argument('--station_fc_units_num', type = int, default=1, help ='the number of station fc layer units')
#
#
#
# parser.add_argument('--lr',type = float, default=0.01, help = 'the learning rate of optimizer')
# parser.add_argument('--step_num',type = int, default=7, help='the number of step')
# parser.add_argument('--batch_num',type = int, default= 3000, help = 'the number of batch')

# args = parser.parse_args()



class Args(object):
    def __init__(self):
        self.path_prefix_data = '../data'
        self.path_predix_summary = '../single_summary_3'

        self.feature_num = 10
        self.station_num = 21

        self.lstm_num_units = 128
        self.local_fc_1_units_num = 32
        self.local_fc_2_units_num = 1
        self.fc_1_units_num = 21

        self.step_num = 700
        self.batch_num = 10000

        self.lr = 0.01
        self.decay_steps = 30
        self.decay_rate = 0.9
        self.clip_grads_max = 30

        self.lstm_input_prob = 1
        self.lstm_output_prob = 0.5 #64
        self.lstm_state_prob = 0.5 #
        self.local_fc_1_prob = 1 #8
        self.local_fc_2_prob = 1 #8
        self.fc1_prob = 1 # 21 * 8 * 0.5

args = Args()


days_info = '天数'

station_info = ['门头沟','房山','云岗','北部新区','古城','植物园'
,'丰台花园','万柳','官园','西直门北','万寿西宫','南三环','奥体中心','前门'
,'永定门内','天坛','大兴','东四','农展馆','东四环','亦庄']


station_geo = [
    ['门头沟',	116.106	,39.937],
['房山'	,116.136	,39.742],
['云岗'	,116.146	,39.824],
['北部新区'	,116.174	,40.09],
['古城'	,116.184	,39.914],
['植物园'	,116.207	,40.002],
['丰台花园'	,116.279	,39.863],
['万柳'	,116.287	,39.987],
['官园'	,116.339	,39.929],
['西直门北'	,116.349	,39.954],
['万寿西宫'	,116.352	,39.878],
['南三环'	,116.368	,39.856],
['奥体中心'	,116.397	,39.982],
['前门'	,116.395	,39.899],
['永定门内'	,116.394	,39.876],
['天坛'	,116.407	,39.886],
['大兴'	,116.404	,39.718],
['东四'	,116.417	,39.929],
['农展馆'	,116.461	,39.937],
['东四环'	,116.483	,39.939],
['亦庄'	,116.506	,39.795],
]

factor_file_name_list = [
    'transIrad_bj.xlsx','transSHum_bj.xlsx','transSRad_bj.xlsx',
    'transprec_bj.xlsx',
                    'transpres_bj.xlsx','transtemp_bj.xlsx','transwind_bj.xlsx']

factor_file_name_dict = {
    'irad':'transIrad_bj.xlsx',
    'shum':'transSHum_bj.xlsx',
    'srad':'transSRad_bj.xlsx',
    'prec':'transprec_bj.xlsx',
    'pres':'transpres_bj.xlsx',
    'temp':'transtemp_bj.xlsx',
    'wind':'transwind_bj.xlsx',
}


pm25_file_name = 'pm25.xlsx'
