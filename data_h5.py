# -*- coding: utf-8 -*-
import sys
import os
import re

os_path = os.path.abspath('./')
find_path = re.compile('emr_electrolyte_abnormal')
BASE_PATH = os_path[:find_path.search(os_path).span()[1]]
sys.path.append(BASE_PATH)

from config import *

import preprocess.preprocess_labtest as lab
import preprocess.preprocess_diagnosis as diag
import preprocess.preprocess_prescribe as pres
import preprocess.preprocess_label as label
import preprocess.preprocess_demographic as demo

from multiprocessing import Pool
from sklearn.model_selection import train_test_split

import time
import tensorflow as tf
import datetime
import random

from functools import partial

import json
import pickle
import h5py
import numpy as np

# 사용할　lab test의　종류
lab_store = pd.HDFStore(LABTEST_PATH, mode='r')
try:
    usecol = lab_store.select('metadata/usecol')
finally:
    lab_store.close()
top_use_col = usecol.sort_values('counts', ascending=False)[
    :NUMS_LABTEST].index

label_store = pd.HDFStore(LABEL_PATH, mode='r')
try:
    na_label_df = label_store.select('label/L3041')
    ka_label_df = label_store.select('label/L3042')

    na_label_ts_df = label_store.select('RNN/label/L3041')
    ka_label_ts_df = label_store.select('RNN/label/L3042')

    test_no = label_store.select('split/L3042/test')
    train_no = label_store.select('split/L3042/train')
    validation_no = label_store.select('split/L3042/validation')
finally:
    label_store.close()

# 데이터 셋을 구분
train_df = ka_label_ts_df[ka_label_ts_df.no.isin(train_no.no)]
# 항상　정상수치가　나온　경우
normal_df = train_df[(train_df.hyper_count == 0) & (train_df.hypo_count == 0)]
# hyper가 나온 경우
hyper_df = train_df[(train_df.hyper_count > 0) & (train_df.hypo_count == 0)]
# hypo가 나온 경우
hypo_df = train_df[(train_df.hyper_count == 0) & (train_df.hypo_count > 0)]
# 둘다 나온 경우
both_df = train_df[(train_df.hyper_count > 0) & (train_df.hypo_count > 0)]

valid_df = ka_label_ts_df[ka_label_ts_df.no.isin(validation_no.no)]


def get_lab_ts_df_with_missing(no, start_date, end_date, use_col):
    '''
    return lab_test timeserial dataframe per patient
    arguments
        use_col : the index name being used in dataframe
        start_date ~ end_date : the time-serial length in dataframe
    '''
    lab_ts_df = lab.get_timeserial_lab_df(no).loc[:, start_date:end_date].\
        reindex(index=use_col, columns=pd.date_range(
            start_date, end_date, freq='D'))
    missing_df = lab_ts_df.isnull().astype(float)
    lab_ts_df = lab_ts_df.fillna(0)
    return pd.concat([lab_ts_df, missing_df])


def get_label_ts_df(label_df, no, start_date, end_date):
    # get the timeserial dataframe of patient's label
    ts_label_df = label_df[(label_df.no == no) & (
        label_df.date >= start_date) & (label_df.date <= end_date)]
    ts_label_df = ts_label_df.drop_duplicates(['no', 'date'], keep='last')

    per_label_df = ts_label_df[['date', 'label']]\
        .pivot_table(columns=['date'], values='label')\
        .reindex(columns=pd.date_range(start_date, start_date+np.timedelta64(SEQ_LENGTH, 'D'), freq='D'))

    label_np = per_label_df.iloc[0].values
    return label_np


def roll_back_label(label_np, roll_back_nums):
    if roll_back_nums == 0:
        pre_label_np = label_np

    while roll_back_nums > 0:
        pre_label_np = np.pad(label_np, 1, 'constant')[2:]
        stacked_label_np = np.stack([label_np, pre_label_np])
        pre_label_np = np.apply_along_axis(
            determine_value, 0, stacked_label_np)
        roll_back_nums = roll_back_nums - 1

    pre_label_np = np.nan_to_num(pre_label_np)
    return pre_label_np


def determine_value(label_np):
    if np.isnan(label_np[0]) or label_np[0] == 0:
        if np.isnan(label_np[1]):
            return label_np[0]
        return label_np[1]
    else:
        return label_np[0]


def _apply_weight_dict(array):
    result = array.copy()
    for key, value in WEIGHT_DICT.items():
        result[array == key] = value
    result[np.isnan(array)] = 0
    return result


def _get_existence_array(array):
    result = array.copy()
    return (~np.isnan(result)).astype(np.int64)


def get_input_lab_per_(row, roll_back_nums):
    global SEQ_LENGTH, WEIGHT_DICT, top_use_col, MIN_DATE
    seq_length = (row.end_date - row.start_date).days

    disease_history_date = row.start_date-np.timedelta64(180, 'D')
    if disease_history_date <= np.datetime64(MIN_DATE[:4]+"-"+MIN_DATE[4:6]+"-"+MIN_DATE[6:]):
        disease_history_date = np.datetime64(
            MIN_DATE[:4]+"-"+MIN_DATE[4:6]+"-"+MIN_DATE[6:])

    if seq_length >= SEQ_LENGTH:
        seq_length = SEQ_LENGTH
    r_end_date = row.start_date+np.timedelta64(SEQ_LENGTH, 'D')

    ka_label_np = get_label_ts_df(
        ka_label_df, row.no, row.start_date, row.end_date)
    na_label_np = get_label_ts_df(
        na_label_df, row.no, row.start_date, row.end_date)

    #ka_exist_np = _apply_weight_dict(ka_label_np)
    #na_exist_np = _apply_weight_dict(na_label_np)

    ka_exist_np = _get_existence_array(ka_label_np)
    na_exist_np = _get_existence_array(na_label_np)

    ka_label_np = roll_back_label(ka_label_np, roll_back_nums)
    na_label_np = roll_back_label(na_label_np, roll_back_nums)

    lab_np = get_lab_ts_df_with_missing(
        row.no, row.start_date, r_end_date, top_use_col).values
    demo_np = demo.get_timeserial_demographic(row.no).reindex(
        columns=pd.date_range(row.start_date, r_end_date, freq='D')).values

    diag_np = diag.get_timeserial_diagnosis_df(row.no, fill_na=False).\
        reindex(columns=pd.date_range(disease_history_date, r_end_date, freq='D')).fillna(axis=1, method='ffill').fillna(0).\
        reindex(columns=pd.date_range(
            row.start_date, r_end_date, freq='D')).values

    pres_np = pres.get_timeserial_prescribe_df(row.no).reindex(
        columns=pd.date_range(row.start_date, r_end_date, freq='D')).values

    return na_label_np, ka_label_np, seq_length, lab_np.T, demo_np.T, diag_np.T, pres_np.T, na_exist_np, ka_exist_np


def get_rnn_input_dataset(df, roll_back_nums):
    global CORE_NUMS
    global result
    pool = Pool()
    result = pool.map_async(
        partial(_get_rnn_input_dataset, roll_back_nums), np.array_split(df, CORE_NUMS))
    concat_na_label = np.concatenate([na_label_np for na_label_np, ka_label_np, seq_length,
                                      lab_np, demo_np, diag_np, pres_np, na_exist_np, ka_exist_np in result.get()])
    concat_ka_label = np.concatenate([ka_label_np for na_label_np, ka_label_np, seq_length,
                                      lab_np, demo_np, diag_np, pres_np, na_exist_np, ka_exist_np in result.get()])
    concat_length = np.concatenate([seq_length for na_label_np, ka_label_np, seq_length,
                                    lab_np, demo_np, diag_np, pres_np, na_exist_np, ka_exist_np in result.get()])
    concat_lab = np.concatenate([lab_np for na_label_np, ka_label_np, seq_length,
                                 lab_np, demo_np, diag_np, pres_np, na_exist_np, ka_exist_np in result.get()])
    concat_demo = np.concatenate([demo_np for na_label_np, ka_label_np, seq_length,
                                  lab_np, demo_np, diag_np, pres_np, na_exist_np, ka_exist_np in result.get()])
    concat_diag = np.concatenate([diag_np for na_label_np, ka_label_np, seq_length,
                                  lab_np, demo_np, diag_np, pres_np, na_exist_np, ka_exist_np in result.get()])
    concat_pres = np.concatenate([pres_np for na_label_np, ka_label_np, seq_length,
                                  lab_np, demo_np, diag_np, pres_np, na_exist_np, ka_exist_np in result.get()])
    concat_na_exist = np.concatenate([na_exist_np for na_label_np, ka_label_np, seq_length,
                                      lab_np, demo_np, diag_np, pres_np, na_exist_np, ka_exist_np in result.get()])
    concat_ka_exist = np.concatenate([ka_exist_np for na_label_np, ka_label_np, seq_length,
                                      lab_np, demo_np, diag_np, pres_np, na_exist_np, ka_exist_np in result.get()])

    return concat_na_label, concat_ka_label, concat_length, concat_lab, concat_demo, concat_diag, concat_pres, concat_na_exist, concat_ka_exist


def _get_rnn_input_dataset(roll_back_nums, df):
    global SEQ_LENGTH
    na_label_list = []
    ka_label_list = []
    len_list = []
    lab_list = []
    demo_list = []
    diag_list = []
    pres_list = []
    ka_exist_list = []
    na_exist_list = []

    for _, row in df.iterrows():
        if (row.end_date - row.start_date) <= np.timedelta64(SEQ_LENGTH, 'D'):
            _na_label, _ka_label, _len, _lab, _demo, _diag, _pres, na_exist, ka_exist = get_input_lab_per_(
                row, roll_back_nums)

            na_label_list.append(_na_label)
            ka_label_list.append(_ka_label)
            len_list.append(_len)
            lab_list.append(_lab)
            demo_list.append(_demo)
            diag_list.append(_diag)
            pres_list.append(_pres)
            na_exist_list.append(na_exist)
            ka_exist_list.append(ka_exist)
        else:
            temp_df = ka_label_df[(ka_label_df.no == row.no) & (
                ka_label_df.date >= row.start_date) & (ka_label_df.date <= row.end_date)]
            temp_row = row.copy()
            for _, inner_row in temp_df.iterrows():
                x = temp_df[(temp_df.date >= inner_row.date +
                             np.timedelta64(SEQ_LENGTH, 'D'))]
                if x.empty:
                    break
                else:
                    temp_row['start_date'] = inner_row.date
                    temp_row['end_date'] = x.iloc[0].date
                    _na_label, _ka_label, _len, _lab, _demo, _diag, _pres, na_exist, ka_exist = get_input_lab_per_(
                        temp_row, roll_back_nums)
                    na_label_list.append(_na_label)
                    ka_label_list.append(_ka_label)
                    len_list.append(_len)
                    lab_list.append(_lab)
                    demo_list.append(_demo)
                    diag_list.append(_diag)
                    pres_list.append(_pres)
                    na_exist_list.append(na_exist)
                    ka_exist_list.append(ka_exist)

    return np.stack(na_label_list), np.stack(ka_label_list), np.stack(len_list),\
        np.stack(lab_list), np.stack(demo_list), np.stack(diag_list),\
        np.stack(pres_list), np.stack(na_exist_list), np.stack(ka_exist_list)


def get_dataset(df, roll_back_nums, array_name, data_name='dataset'):
    dataset_name = "{}_{}".format(data_name, roll_back_nums)
    node_name = dataset_name + "/" + array_name
    h5f = h5py.File(H5_DATASET_PATH, 'r')
    if "{}_{}/{}".format(data_name, roll_back_nums, array_name) in h5f:
        h5f.close()
        print("{} exists.")
        dataset = read_in_h5f(dataset_name, array_name)
    else:
        print("{} doesn't exist. start to construct".format(node_name))
        h5f.close()
        start_time = time.time()
        concat_na_label, concat_ka_label, concat_length, concat_lab, concat_demo, concat_diag, concat_pres, concat_na_exist, concat_ka_exist =\
            get_rnn_input_dataset(df, roll_back_nums)
        print("time consumed --{}".format(time.time()-start_time))
        dataset = list(zip(concat_na_label, concat_ka_label, concat_length, concat_lab,
                           concat_demo, concat_diag, concat_pres, concat_na_exist, concat_ka_exist))
        write_in_h5f(dataset_name, array_name, dataset)
    return dataset


def write_in_h5f(dataset_name, array_name, dataset):
    na_label, ka_label, length_, lab_, demo_, diag_, pres_, na_exist, ka_exist = list(
        zip(*dataset))

    na_label_np = np.stack(na_label).astype(float)
    ka_label_np = np.stack(ka_label).astype(float)
    length_np = np.stack(length_).astype(float)
    lab_np = np.stack(lab_).astype(float)
    demo_np = np.stack(demo_).astype(float)
    diag_np = np.stack(diag_).astype(float)
    pres_np = np.stack(pres_).astype(float)
    na_exist_np = np.stack(na_exist).astype(float)
    ka_exist_np = np.stack(ka_exist).astype(float)

    node_name = dataset_name + "/" + array_name
    with h5py.File(H5_DATASET_PATH, 'a') as h5f:
        h5f.create_dataset(node_name+"/na_label", data=na_label_np)
        h5f.create_dataset(node_name+"/ka_label", data=ka_label_np)
        h5f.create_dataset(node_name+"/length", data=length_np)
        h5f.create_dataset(node_name+"/lab", data=lab_np)
        h5f.create_dataset(node_name+"/demo", data=demo_np)
        h5f.create_dataset(node_name+"/diag", data=diag_np)
        h5f.create_dataset(node_name+"/pres", data=pres_np)
        h5f.create_dataset(node_name+"/na_exist", data=na_exist_np)
        h5f.create_dataset(node_name+"/ka_exist", data=ka_exist_np)


def read_in_h5f(dataset_name, array_name):
    node_name = dataset_name + '/' + array_name
    h5f = h5py.File(H5_DATASET_PATH, 'r')
    try:
        if not node_name in h5f:
            print("need to write {}".format(node_name))
    finally:
        h5f.close()

    with h5py.File(H5_DATASET_PATH, 'r') as h5f:
        na_label_np = h5f[node_name+'/na_label'][:]
        ka_label_np = h5f[node_name+"/ka_label"][:]
        length_np = h5f[node_name+'/length'][:]
        lab_np = h5f[node_name+'/lab'][:]
        demo_np = h5f[node_name+"/demo"][:]
        diag_np = h5f[node_name+'/diag'][:]
        pres_np = h5f[node_name+'/pres'][:]
        na_exist_np = h5f[node_name+'/na_exist'][:]
        ka_exist_np = h5f[node_name+'/ka_exist'][:]
        return list(zip(na_label_np, ka_label_np, length_np, lab_np, demo_np, diag_np, pres_np, na_exist_np, ka_exist_np))


def dataset_batch_generator(dataset_name, array_name, batch_num):
    dataset = read_in_h5f(dataset_name, array_name)
    dataset_size = len(dataset)
    random.shuffle(dataset)
    batch_index = 0
    while True:
        na_label_list = []
        ka_label_list = []
        length_list = []
        lab_list = []
        demo_list = []
        diag_list = []
        pres_list = []
        na_exist_list = []
        ka_exist_list = []

        if dataset_size > batch_index + batch_num:
            random.shuffle(dataset)
            batch_index = 0

        for idx in range(batch_index, batch_index+batch_num):
            na_label, ka_label, _length, _lab, _demo, _diag, _pres, na_exist, ka_exist = dataset[idx]
            na_label_list.append(na_label)
            ka_label_list.append(ka_label)
            length_list.append(_length)
            lab_list.append(_lab)
            demo_list.append(_demo)
            diag_list.append(_diag)
            pres_list.append(_pres)
            na_exist_list.append(na_exist)
            ka_exist_list.append(ka_exist)

        batch_index = batch_index+batch_num
        yield np.stack(na_label_list), np.stack(ka_label_list), np.stack(length_list),\
            np.stack(lab_list), np.stack(demo_list), np.stack(diag_list),\
            np.stack(pres_list), np.stack(na_exist_list), np.stack(ka_exist_list)
