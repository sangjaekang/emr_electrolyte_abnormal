# -*- coding: utf-8 -*-
import sys, os, re

os_path = os.path.abspath('./') ; find_path = re.compile('emr_electrolyte_abnormal')
BASE_PATH = os_path[:find_path.search(os_path).span()[1]]
sys.path.append(BASE_PATH)

from config import *
import preprocess.preprocess_labtest as lab
'''
LABEL DATA I/O 관련 모듈
labtest data에서　환자의　전해질이상　case dataframe을　출력하는　함수

핵심 메소드
    * get_timeserial_lab_label(no)
        환자의　전해질이상　case dataframe을　출력하는　함수
        ----
        output matrix property
            columns : no lab_test date result label
            row : case
'''
def get_timeserial_lab_label(no,lab_test='L3042'):
    global LABTEST_PATH, SKIP_LABEL_INTERVAL
    # 전해질코드에 따른 환자의 전해질 이상 label dataframe을 출력하는 함수

    #전처리된 데이터가 없으면 전처리하여 생성
    if not os.path.isfile(LABTEST_PATH):
        if DEBUG_PRINT: print("no LABTEST file")
        lab.preprocess_labtest()

    lab_store = pd.HDFStore(LABTEST_PATH,mode='r')
    try:
        target_df = lab_store.select('original',where='no=={} & lab_test=="{}"'.format(no,lab_test))
    finally:
        lab_store.close()

    if lab_test == 'L3041':
        convert_label = convert_na_label
    elif lab_test == 'L3042':
        convert_label = convert_ka_label
    else:
        raise ValueError("Wrong lab_test num. there is no convert_label method")
    
    target_df.loc[:,'label'] = target_df.result.map(convert_label)

    label_df = pd.DataFrame(columns=['no','lab_test','date','result','label'])
    for i,row in target_df.iterrows():
        if row.label == 0:
            # 정상인경우， 앞뒤 skip_label_interval 기간동안 전해질 이상이 없으면 인정 아니면 SKIP
            temp_df = target_df[(target_df.date>=row.date-np.timedelta64(SKIP_LABEL_INTERVAL,"D"))\
                                & (target_df.date<row.date+np.timedelta64(SKIP_LABEL_INTERVAL,"D"))\
                                & (target_df.label!=0)]
            if temp_df.empty:
                label_df=label_df.append(row,ignore_index=True)
        elif row.label == 1:
            # 전해질이상인경우， 앞 skip_label_interval 기간동안 다른 전해질 이상이 없으면 인정 아니면 SKIP
            temp_df = target_df[(target_df.date>=row.date-np.timedelta64(SKIP_LABEL_INTERVAL,"D"))\
                                & (target_df.date<row.date)\
                                & (target_df.label==2)]
            if temp_df.empty:
                label_df=label_df.append(row,ignore_index=True)
        else:
            # 전해질이상인경우， 앞 skip_label_interval 기간동안 다른 전해질 이상이 없으면 인정 아니면 SKIP
            temp_df = target_df[(target_df.date>=row.date-np.timedelta64(SKIP_LABEL_INTERVAL,"D"))\
                                & (target_df.date<row.date)\
                                & (target_df.label==1)]
            if temp_df.empty:
                label_df=label_df.append(row,ignore_index=True)
    
    return label_df


def convert_na_label(x):
    if x < 135:
        return 1
    elif x > 145:
        return 2
    else:
        return 0


def convert_ka_label(x):
    if x < 3.5:
        return 1
    elif x > 5.1:
        return 2
    else:
        return 0