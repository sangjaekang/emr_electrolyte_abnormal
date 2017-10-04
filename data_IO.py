# -*- coding: utf-8 -*-
import sys, os, re

os_path = os.path.abspath('./') ; find_path = re.compile('emr_electrolyte_abnormal')
BASE_PATH = os_path[:find_path.search(os_path).span()[1]]
sys.path.append(BASE_PATH)

from config import *

import preprocess.preprocess_labtest as lab
import preprocess.preprocess_diagnosis as diag
import preprocess.preprocess_prescribe as pres
import preprocess.preprocess_label as label
import preprocess.preprocess_demographic as demo

from multiprocessing import Pool

def skip_case(lab_test, diag_counts, pres_counts, lab_counts):
    # diag_counts, pres_counts, lab_counts가 기준치에 미달할 경우
    # 그 case는 모델 학습에서 제외시킴
    # Too sparse를 피하기 위함
    global LABEL_PATH
    label_store = pd.HDFStore(LABEL_PATH,mode='r')
    try:
        label_df = label_store.select("prep/label/{}".format(lab_test))
    finally:
        label_store.close()
    
    if diag_counts is None : diag_cond = True
    else : diag_cond = (label_df.lab_counts >= diag_counts)
    if pres_counts is None : pres_cond = True
    else : pres_cond = (label_df.lab_counts >= pres_counts)
    if lab_counts is None : lab_cond = True
    else : lab_cond = (label_df.lab_counts >= lab_counts)
    
    if isinstance(diag_cond,bool) & isinstance(pres_cond,bool) & isinstance(lab_cond,bool):
        result_df = label_df
    else:
        result_df = label_df[diag_cond & pres_cond & lab_cond]
    
    result_df = result_df.drop_duplicates()
    # label 갯수 확인
    print("* 전체 갯수----")
    for label,counts in result_df.label.value_counts().items():
        print("   {} label - {} counts".format(label,counts))
    
    # 갯수 Count
    label_store.open(mode='r')
    try:
        no_list = label_store.select('split/train').no
        print("* Train 갯수----")
        train_df = result_df[result_df.no.isin(no_list)]
        for label,counts in train_df.label.value_counts().items():
            print("   {} label - {} counts".format(label,counts))
        
        no_list = label_store.select('split/test').no
        print("* test 갯수----")
        train_df = result_df[result_df.no.isin(no_list)]
        for label,counts in train_df.label.value_counts().items():
            print("   {} label - {} counts".format(label,counts))

        no_list = label_store.select('split/validation').no
        print("* Validation 갯수----")
        train_df = result_df[result_df.no.isin(no_list)]
        for label,counts in train_df.label.value_counts().items():
            print("   {} label - {} counts".format(label,counts))
    finally:
        label_store.close()
        
    return result_df

def get_diag_ts_df(no,date):
    global GAP_PERIOD, TARGET_PERIOD
    t_day = date - np.timedelta64(GAP_PERIOD,'D')
    f_day = t_day - np.timedelta64(TARGET_PERIOD-1,'D')
    diag_ts_df = diag.get_timeserial_diagnosis_df(no).loc[:,f_day:t_day]
    # 기간동안 ２번이상 진단코드가 있으면，그 사이를 채움
    code_count = diag_ts_df.sum(1)
    for code, _ in code_count[code_count >1].items():
        ts_series = diag_ts_df.loc[code,:]
        first_code_day = ts_series[ts_series==1].index[0]
        last_code_day = ts_series[ts_series==1].index[-1]
        diag_ts_df.loc[code,first_code_day:last_code_day] = 1.0
    return diag_ts_df

def get_pres_ts_df(no,date):
    global GAP_PERIOD, TARGET_PERIOD
    t_day = date - np.timedelta64(GAP_PERIOD,'D')
    f_day = t_day - np.timedelta64(TARGET_PERIOD-1,'D')
    pres_ts_df = pres.get_timeserial_prescribe_df(no).loc[:,f_day:t_day]
    return pres_ts_df

def get_demo_ts_df(no,date):
    global GAP_PERIOD, TARGET_PERIOD
    t_day = date - np.timedelta64(GAP_PERIOD,'D')
    f_day = t_day - np.timedelta64(TARGET_PERIOD-1,'D')
    demo_ts_df = demo.get_timeserial_demographic(no).loc[:,f_day:t_day]
    return demo_ts_df

def get_lab_ts_df(no,date):
    global GAP_PERIOD, TARGET_PERIOD
    t_day = date - np.timedelta64(GAP_PERIOD,'D')
    f_day = t_day - np.timedelta64(TARGET_PERIOD-1,'D')
    lab_ts_df = lab.get_timeserial_lab_df(no).loc[:,f_day:t_day]
    return lab_ts_df

def get_patient_ts_df(no,date):
    global GAP_PERIOD, TARGET_PERIOD
    demo_df = get_demo_ts_df(no,date)
    pres_df   = get_pres_ts_df(no,date)
    diag_df   = get_diag_ts_df(no,date)
    lab_df     = get_lab_ts_df(no,date)
    return pd.concat([demo_df,pres_df,diag_df,lab_df])
