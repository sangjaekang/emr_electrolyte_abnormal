# -*- coding: utf-8 -*-
%matplotlib inline
import sys, os, re
import pandas as pd
import numpy as np
import os

os_path = os.path.abspath('./') ; find_path = re.compile('emr_electrolyte_abnormal')
BASE_PATH = os_path[:find_path.search(os_path).span()[1]]
sys.path.append(BASE_PATH)

from config import *

import preprocess.preprocess_labtest as lab
import preprocess.preprocess_diagnosis as diag
import preprocess.preprocess_prescribe as pres
import preprocess.preprocess_label as label

from multiprocessing import Pool
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy

'''
FEATURE SELECTION 모듈
진단데이터와처방데이터는　매우　sparse한　데이터
모든　진단코드와　처방코드를　쓰기에는
핵심　패턴을　도출하기에　데이터　양이　적다고　판단

Boruta는　다양한　feature　중에서　핵심　feature을　찾아주는　메소드

Boruta 알고리즘 (ref : http://www.listendata.com/2017/05/feature-selection-boruta-package.html)
Follow the steps below to understand the algorithm

1. Create duplicate copies of all independent variables. 
    When the number of independent variables in the original data is less than 5,  
    create at least 5 copies using existing variables.

2. Shuffle the values of added duplicate copies 
    to remove their correlations with the target variable. 
    It is called shadow features or permuted copies.

3. Combine the original ones with shuffled copies

4. Run a random forest classifier on the combined dataset 
    and performs a variable importance measure 
    (the default is Mean Decrease Accuracy) to evaluate the importance of each variable where higher means more important.

5. Then Z score is computed. 
    It means mean of accuracy loss divided by standard deviation of accuracy loss.

6. Find the maximum Z score among shadow attributes (MZSA)

7. Tag the variables as 'unimportant'  
    when they have importance significantly lower than MZSA. Then we permanently remove them from the process.

8. Tag the variables as 'important'  
    when they have importance significantly higher than MZSA.

9.Repeat the above steps for predefined number of iterations (random forest runs), 
    or until all attributes are either tagged 'unimportant' or 'important', whichever comes first.

    
'''

def get_timeserial_lab_label(no,lab_test='L3042'):
    global LABTEST_PATH, SKIP_LABEL_INTERVAL
    # 전해질코드에 따른 환자의 전해질 이상 label dataframe을 출력하는 함수
    # feature selection에 맞게 약간 수정함
    # 수정 내용 ：같은 전해질 이상이 앞에 있어도 SKIP

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
        convert_label = label.convert_na_label
    elif lab_test == 'L3042':
        convert_label = label.convert_ka_label
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
            temp_df = target_df[(target_df.date>=row.date-np.timedelta64(2*SKIP_LABEL_INTERVAL,"D"))\
                                & (target_df.date<row.date)\
                                & (target_df.label!=0)] #수정 내용
            if temp_df.empty:
                label_df=label_df.append(row,ignore_index=True)
        else:
            # 전해질이상인경우， 앞 skip_label_interval 기간동안 다른 전해질 이상이 없으면 인정 아니면 SKIP
            temp_df = target_df[(target_df.date>=row.date-np.timedelta64(2*SKIP_LABEL_INTERVAL,"D"))\
                                & (target_df.date<row.date)\
                                & (target_df.label!=0)] #수정 내용
            if temp_df.empty:
                label_df=label_df.append(row,ignore_index=True)
    
    return label_df

def get_feature_selection_prescribe(no_list,lab_test='L3042'):        
    result_list = []
    for no in no_list:
        target_df = get_timeserial_lab_label(no,lab_test)
        if target_df.empty:
            continue
        df = pd.DataFrame()
        first = True
        for _, row in target_df.iterrows():
            x = pres.get_timeserial_prescribe_df(no).\
                loc[:,row.date-np.timedelta64(89,'D'):row.date].sum(axis=1)
            if x.sum()>1:
                x = x.set_value('result',row.label)
                if first:
                    df = pd.DataFrame(columns=x.index);first = False
                df = df.append(x,ignore_index=True)
        if not df.empty:
            result_list.append(df)
        if len(result_list) > 0:
            return pd.concat(result_list)


def get_feature_selection_diagnosis(no_list,lab_test='L3042'):        
    result_list = []
    for no in no_list:
        target_df = get_timeserial_lab_label(no,lab_test)
        if target_df.empty:
            continue
        df = pd.DataFrame()
        first = True
        for _, row in target_df.iterrows():
            x = diag.get_timeserial_diagnosis_df(no).\
                loc[:,row.date-np.timedelta64(89,'D'):row.date].sum(axis=1)
            if x.sum()>1:
                x = x.set_value('result',row.label)
                if first: 
                    df = pd.DataFrame(columns=x.index);first = False
                df = df.append(x,ignore_index=True)
        
        if not df.empty: 
            result_list.append(df)
     
     if len(result_list) > 0: 
        return pd.concat(result_list)


def execute_feature_selection_diagnosis(lab_test='L3042'):
    global FEATURE_DIAGNOSIS_PATH, LABTEST_PATH
    pool = Pool()   
    lab_store = pd.HDFStore(LABTEST_PATH,mode='r')
    try:
        no_list = lab_store.select('prep',columns=['no']).no.unique()
    finally:
        lab_store.close()

    result = pool.map_async(get_KCD_column, np.array_split(no_list, 8))
    result_df = pd.concat([x for x in result.get() if x is not None])
    result_df.to_hdf(FEATURE_DIAGNOSIS_PATH,'prep/{}'.format(LAB_CODE),format='table',data_columns=True,mode='a')

    rf = RandomForestClassifier(n_jobs=-1, class_weight='auto', max_depth=5)
    feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)

    normal_df = result_df[result_df.result ==0.0].sample(frac=0.1);
    hypo_df  = result_df[result_df.result ==1.0];
    hyper_df = result_df[result_df.result ==2.0];
    concat_df = pd.concat([normal_df,hypo_df,hyper_df]);

    y = concat_df.pop('result')
    x = concat_df.copy()

    feat_selector.fit(x.as_matrix(),y.as_matrix())
    code = x.columns[feat_selector.support_]
    pd.DataFrame(data=code.values,columns=['code']).to_hdf(FEATURE_DIAGNOSIS_PATH,'usecol/{}'.format(LAB_CODE),format='table',data_columns=True,mode='a')


def execute_feature_selection_prescribe(lab_test='L3042'):
    global FEATURE_PRESCRIBE_PATH, LABTEST_PATH
    pool = Pool()   
    lab_store = pd.HDFStore(LABTEST_PATH,mode='r')
    try:
        no_list = lab_store.select('prep',columns=['no']).no.unique()
    finally:
        lab_store.close()

    result = pool.map_async(get_KCD_column, np.array_split(no_list, 8))
    result_df = pd.concat([x for x in result.get() if x is not None])
    result_df.to_hdf(FEATURE_PRESCRIBE_PATH,'prep/{}'.format(LAB_CODE),format='table',data_columns=True,mode='a')

    rf = RandomForestClassifier(n_jobs=-1, class_weight='auto', max_depth=5)
    feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)

    normal_df = result_df[result_df.result ==0.0].sample(frac=0.1);
    hypo_df  = result_df[result_df.result ==1.0];
    hyper_df = result_df[result_df.result ==2.0];
    concat_df = pd.concat([normal_df,hypo_df,hyper_df]);

    y = concat_df.pop('result')
    x = concat_df.copy()

    feat_selector.fit(x.as_matrix(),y.as_matrix())
    code = x.columns[feat_selector.support_]
    pd.DataFrame(data=code.values,columns=['code']).to_hdf(FEATURE_PRESCRIBE_PATH,'usecol/{}'.format(LAB_CODE),format='table',data_columns=True,mode='a')