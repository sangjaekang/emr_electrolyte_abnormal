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

from multiprocessing import Pool
'''
LABEL DATA I/O 관련 모듈
labtest data에서 환자의 전해질이상 case dataframe을 출력하는 함수

핵심 메소드
    * get_timeserial_lab_label(no)
        환자의 전해질이상 case dataframe을 출력하는 함수
        ----
        output matrix property
            columns : no lab_test date result label
            row : case
    * preprocess_label(lab_test)
        lab_test code 기준으로 환자의 전해질이상에 대한 labeling을 하는 함수

    * split_train_validation_test()
        환자 번호 기준으로 train-validation-split을 나누는 함수
'''


def get_timeserial_lab_label(no, lab_test='L3042', axis=1):
    global LABEL_PATH, DEBUG_PRINT, MIN_DATE, MAX_DATE
    # 전해질코드에 따른 환자의 전해질 이상 label dataframe을 출력하는 함수

    #전처리된 데이터가 없으면 전처리하여 생성
    if not os.path.isfile(LABEL_PATH):
        if DEBUG_PRINT:
            print("no LABEL file")
        preprocess_label(lab_test)

    label_store = pd.HDFStore(LABEL_PATH, mode='r')
    try:
        target_df = label_store.select('label/{}'.format(lab_test), where='no=={} '.format(no))
    finally:
        label_store.close()
    if axis == 1:
        return target_df[['date', 'label']]
    else:
        return target_df[['no', 'date', 'label']]\
           .pivot_table(value=['label'], columns=['date'])\
           .reindex(columns=pd.date_range(MIN_DATE, MAX_DATE, freq='D'))


def preprocess_label(lab_test):
    global SKIP_LABEL_INTERVAL, LABTEST_PATH, LABEL_PATH, DEBUG_PRINT, CORE_NUMS
    if DEBUG_PRINT:
        print("preprocess_label starts")

    #전처리된 데이터가 없으면 전처리하여 생성
    if not os.path.isfile(LABTEST_PATH):
        if DEBUG_PRINT:
            print("no LABTEST file")
        lab.preprocess_labtest()

    lab_store = pd.HDFStore(LABTEST_PATH, mode='r')
    try:
        original_df = lab_store.select('original', where='lab_test=="{}"'.format(lab_test))
    finally:
        lab_store.close()

    no_list = original_df.no.unique()
    pool = Pool()
    result = pool.map_async(_preprocess_label, np.array_split(no_list, CORE_NUMS))
    concat_df = pd.concat([x for x in result.get() if x is not None])

    concat_df.no = concat_df.no.astype(int)
    concat_df.label = concat_df.label.astype(int)

    concat_df[['no', 'date', 'label']].to_hdf(LABEL_PATH, 'label/{}'.format(lab_test), format='table', data_columns=True, mode='a')
    if DEBUG_PRINT:
        print("preprocess_label ends")
    split_train_validation_test(lab_test)


def _preprocess_label(no_list, lab_test='L3042'):
    global SKIP_LABEL_INTERVAL, LABTEST_PATH, LABEL_PATH, DEBUG_PRINT
    if DEBUG_PRINT:
        print("  _preprocess_label starts")
    lab_store = pd.HDFStore(LABTEST_PATH, mode='r')
    try:
        original_df = lab_store.select('original', where='lab_test=="{}"'.format(lab_test))
    finally:
        lab_store.close()

    if lab_test == 'L3041':
        convert_label = convert_na_label
    elif lab_test == 'L3042':
        convert_label = convert_ka_label
    else:
        raise ValueError("Wrong lab_test num. there is no convert_label method")
    
    original_df.loc[:, 'label'] = original_df.result.map(convert_label)

    label_list = []
    for no in no_list:
        target_df = original_df[original_df.no == no]
        label_df = pd.DataFrame(columns=['no','lab_test','date','result','label'])
        for i,row in target_df.iterrows():
            if row.label == 0:
                # 정상인경우， 앞뒤 skip_label_interval 기간동안 전해질 이상이 없으면 인정 아니면 SKIP
                temp_df = target_df[(target_df.date >= row.date-np.timedelta64(SKIP_LABEL_INTERVAL, "D"))\
                                    & (target_df.date < row.date+np.timedelta64(SKIP_LABEL_INTERVAL, "D"))\
                                    & (target_df.label != 0)]
                if temp_df.empty:
                    label_df = label_df.append(row, ignore_index=True)
            elif row.label == 1:
                # 전해질이상인경우， 앞 skip_label_interval 기간동안 다른 전해질 이상이 없으면 인정 아니면 SKIP
                temp_df = target_df[(target_df.date >= row.date-np.timedelta64(SKIP_LABEL_INTERVAL, "D"))\
                                    & (target_df.date < row.date)\
                                    & (target_df.label == 2)]
                if temp_df.empty:
                    label_df = label_df.append(row, ignore_index=True)
            else:
                # 전해질이상인경우， 앞 skip_label_interval 기간동안 다른 전해질 이상이 없으면 인정 아니면 SKIP
                temp_df = target_df[(target_df.date >= row.date-np.timedelta64(SKIP_LABEL_INTERVAL, "D"))\
                                    & (target_df.date < row.date)\
                                    & ((target_df.label == 1) & (target_df.label == 3)) ]
                if temp_df.empty:
                    label_df = label_df.append(row, ignore_index=True)
        label_list.append(label_df)

    concat_df = pd.concat(label_list)
    concat_df.no = concat_df.no.astype(int)
    concat_df.label = concat_df.label.astype(int)

    if DEBUG_PRINT:
        print("  _preprocess_label ends")
    return concat_df


def split_train_validation_test(lab_test):
    global DEBUG_PRINT, LABEL_PATH
    if DEBUG_PRINT:
        print("split_train_validation_test starts")
    label_store = pd.HDFStore(LABEL_PATH, mode='r')
    try:
        target_df = label_store.select('label/{}'.format(lab_test))
    finally:
        label_store.close()

    # (ratio) : train - validation - test  = 0.7 : 0.1 : 0.2
    no_list = target_df.no.unique()
    train_no, test_no = train_test_split(no_list, test_size=0.3)
    test_no, validation_no = train_test_split(test_no, test_size=0.33)

    train_no.sort()
    test_no.sort()
    validation_no.sort()

    pd.DataFrame(data=train_no, columns=['no']).to_hdf(LABEL_PATH, 'split/train', format='table', data_columns=True, mode='a')
    pd.DataFrame(data=test_no, columns=['no']).to_hdf(LABEL_PATH, 'split/test', format='table', data_columns=True, mode='a')
    pd.DataFrame(data=validation_no, columns=['no']).to_hdf(LABEL_PATH, 'split/validation', format='table', data_columns=True, mode='a')
    if DEBUG_PRINT:
        print("split_train_validation_test ends")


def get_not_sparse_data(lab_test):
    '''
    data중에서　지나치게　sparse한　데이터가　존재하는　경우，
    학습에　어려움이　있음
    미리　지나치게　sparse한　데이터인　경우들을　체크해두고，
    이를　dataset에서　포함시키지　않고자　함
    '''
    global CORE_NUMS, DEBUG_PRINT, LABEL_PATH, GAP_PERIOD, TARGET_PERIOD
    
    label_store = pd.HDFStore(LABEL_PATH, mode='r')
    if DEBUG_PRINT:
        print("get_not_sparse_data starts")
    try:
        label_df = label_store.select('label/{}'.format(lab_test))
    finally:
        label_store.close()
    
    no_list = label_df.no.unique()
    pool = Pool()
    result = pool.map_async(_get_not_sparse_data, np.array_split(no_list, CORE_NUMS))
    result_df = pd.concat([x for x in result.get() if x is not None])
    result_df.no = result_df.no.astype(int)
    result_df.label = result_df.label.astype(int)
    result_df.diag_counts = result_df.diag_counts.astype(int)
    result_df.pres_counts = result_df.pres_counts.astype(int)
    result_df.lab_counts = result_df.lab_counts.astype(int)
    result_df.to_hdf(LABEL_PATH, 'prep/label/{}'.format(lab_test), format='table', data_columns=True, mode='a')
    # save the metadata about the this dataset
    metadata_df = pd.DataFrame(index=['GAP_PERIOD', 'TARGET_PERIOD'], columns=['result'])
    metadata_df.loc['GAP_PERIOD' ,'result'] = GAP_PERIOD
    metadata_df.loc['TARGET_PERIOD' ,'result'] = TARGET_PERIOD
    metadata_df.result = metadata_df.result.astype(int)
    metadata_df.to_hdf(LABEL_PATH, '/metadata/{}'.format(lab_test), format='table', data_columns=True, mode='a')
    
    if DEBUG_PRINT:
        print("get_not_sparse_data ends")


def _get_not_sparse_data(no_list):
    global DEBUG_PRINT, LABEL_PATH, MIN_DATE, GAP_PERIOD, TARGET_PERIOD, PREDICTION_PERIOD

    if DEBUG_PRINT: print("_get_not_sparse_data starts")
    label_store = pd.HDFStore(LABEL_PATH,mode='r')
    try:
        label_df = label_store.select('label/L3042')
    finally:
        label_store.close()
    result_df = pd.DataFrame(columns=['no','date','first_date','last_date','diag_counts','pres_counts','lab_counts','label'])
    for no in no_list:
        diag_ts_df = diag.get_timeserial_diagnosis_df(no)
        pres_ts_df = pres.get_timeserial_prescribe_df(no)
        lab_ts_df  = lab.get_timeserial_lab_df(no)
        for _,row in label_df[label_df.no==no].iterrows():
            #target_period 처음　날짜(t_day)와　끝　날짜(f_day)　
            t_day = row.date- np.timedelta64(GAP_PERIOD,'D')
            f_day = t_day - np.timedelta64(TARGET_PERIOD,'D')
            
            #target_period 사이의　데이터　갯수
            diag_counts = get_df_between_date(diag_ts_df,t_day,f_day)
            pres_counts = get_df_between_date(pres_ts_df,t_day,f_day)
            lab_counts = get_df_between_date_lab(lab_ts_df,t_day,f_day)
            
            # target_period 내 처음과마지막 검사 or 진단 받은 날짜

            #시간　범위가　벗어나면　skip
            if (f_day-np.timedelta64(PREDICTION_PERIOD,'D') >= np.datetime64(MIN_DATE[:4]+'-'+MIN_DATE[4:6]+'-'+MIN_DATE[6:])):
                _row = row.set_value('diag_counts',diag_counts)\
                          .set_value('pres_counts',pres_counts)\
                          .set_value('lab_counts',lab_counts)
                result_df = result_df.append(_row,ignore_index=True)
    
    if DEBUG_PRINT: print("_get_not_sparse_data ends")
    return result_df

def get_df_between_date(df, t_day,f_day):
    return df.loc[:,df.columns[(df.columns > f_day) & (df.columns < t_day)]].sum().sum()

def get_df_between_date_lab(df, t_day,f_day):
    return df.loc[:,df.columns[(df.columns > f_day) & (df.columns < t_day)]].count().sum()

def convert_na_label(x):
    if x < 135:
        return 1
    elif x > 145:
        return 2
    else:
        return 0


def convert_ka_label(x):
    # reference : 
    # 저칼륨혈증과 고칼륨혈증, 임인석
    if x <= 3.0:# 중증저칼륨혈증
        return 1
    elif x >= 5.5:# 중증고칼륨혈증
        return 2
    elif x < 3.5:# 경증저칼륨혈증
        return 3
    elif x > 5.0:# 경증고칼륨혈증
        return 4
    else:# 정상(3.5 ~ 5.0)
        return 0