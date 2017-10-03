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
    * preprocess_label(lab_test)
        lab_test code 기준으로　환자의　전해질이상에　대한　labeling을　하는　함수
'''
def get_timeserial_lab_label(no,lab_test='L3042',axis=1):
    global LABEL_PATH, DEBUG_PRINT
    # 전해질코드에 따른 환자의 전해질 이상 label dataframe을 출력하는 함수

    #전처리된 데이터가 없으면 전처리하여 생성
    if not os.path.isfile(LABEL_PATH):
        if DEBUG_PRINT: print("no LABEL file")
        preprocess_label(lab_test)

    label_store = pd.HDFStore(LABEL_PATH,mode='r')
    try:
        target_df = label_store.select('label/{}'.format(lab_test),where='no=={}'.format(no))
    finally:
        label_store.close()
    if axis==1:
        return target_df[['date','label']]
    else:
        return target_df[['no','date','label']]\
           .pivot_table(value=['label'],columns=['date'])\
           .reindex(columns=pd.date_range(MIN_DATE,MAX_DATE,freq='D'))

def preprocess_label(lab_test):
    global SKIP_LABEL_INTERVAL, LABTEST_PATH, LABEL_PATH, DEBUG_PRINT
    if DEBUG_PRINT: print("preprocess_label starts")

    #전처리된 데이터가 없으면 전처리하여 생성
    if not os.path.isfile(LABTEST_PATH):
        if DEBUG_PRINT: print("no LABTEST file")
        lab.preprocess_labtest()

    lab_store = pd.HDFStore(LABTEST_PATH,mode='r')
    try:
        original_df = lab_store.select('original',where='lab_test=="{}"'.format(lab_test))
    finally:
        lab_store.close()

    if lab_test == 'L3041':
        convert_label = convert_na_label
    elif lab_test == 'L3042':
        convert_label = convert_ka_label
    else:
        raise ValueError("Wrong lab_test num. there is no convert_label method")
    
    original_df.loc[:,'label'] = original_df.result.map(convert_label)

    label_list = []
    for no in original_df.no.unique():
        target_df = original_df[original_df.no == no]
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
        label_list.append(label_df)

    concat_df = pd.concat(label_list)
    concat_df[['no','date','label']].to_hdf(LABEL_PATH,'label/{}'.format(lab_test),format='table',data_columns=True,mode='a')
    if DEBUG_PRINT: print("preprocess_label ends")

def split_train_validation_test():
    global DEBUG_PRINT, LABEL_PATH
    if DEBUG_PRINT: print("split_train_validation_test starts")
    label_store = pd.HDFStore(LABEL_PATH,mode='r')
    try:
        target_df = label_store.select('label/{}'.format(lab_test))
    finally:
        label_store.close()

    # (ratio) : train - validation - test  = 0.7 : 0.1 : 0.2
    no_list = target_df.no.unique()
    train_no, test_no = train_test_split(no_list,test_size=0.3)
    test_no, validation_no = train_test_split(test_no,test_size=0.33)

    train_no.sort()
    test_no.sort()
    validation_no.sort()

    pd.DataFrame(data=train_no,columns=['no']).to_hdf(LABEL_PATH,'split/train',format='table',data_columns=True,mode='a')
    pd.DataFrame(data=test_no,columns=['no']).to_hdf(LABEL_PATH,'split/test',format='table',data_columns=True,mode='a')
    pd.DataFrame(data=validation_no,columns=['no']).to_hdf(LABEL_PATH,'split/validation',format='table',data_columns=True,mode='a')
    if DEBUG_PRINT: print("split_train_validation_test ends")

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