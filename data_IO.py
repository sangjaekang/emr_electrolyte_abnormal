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

def memoize(func):
    cache = {}
    def memoizer(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    return memoizer

@memoize
def skip_case(lab_test, diag_counts=None, pres_counts=None, lab_counts=None,types='train'):
    # diag_counts, pres_counts, lab_counts가 기준치에 미달할 경우
    # 그 case는 모델 학습에서 제외시킴
    # Too sparse를 피하기 위함
    # diag_counts / pres_counts / lab_counts : 각데이터별최소갯수
    # types : total / train / test / validation

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
        test_df = result_df[result_df.no.isin(no_list)]
        for label,counts in test_df.label.value_counts().items():
            print("   {} label - {} counts".format(label,counts))

        no_list = label_store.select('split/validation').no
        print("* Validation 갯수----")
        validation_df = result_df[result_df.no.isin(no_list)]
        for label,counts in validation_df.label.value_counts().items():
            print("   {} label - {} counts".format(label,counts))
    finally:
        label_store.close()
        
    if types=='total':
        return result_df
    elif types=='train':
        return train_df
    elif types=='test':
        return test_df
    elif types=='validation':
        return validation_df
    else :
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

def get_lab_ts_df(no,date,imputation=True):
    global GAP_PERIOD, TARGET_PERIOD
    t_day = date - np.timedelta64(GAP_PERIOD,'D')
    f_day = t_day - np.timedelta64(TARGET_PERIOD-1,'D')
    lab_ts_df = lab.get_timeserial_lab_df(no).loc[:,f_day:t_day]
    if imputation:
        return impute_lab_basic(lab_ts_df)
    else:
        return lab_ts_df

def impute_lab_basic(lab_df):
    lab_array = lab_df.values
    # lab_array는　lab_df의　numpy형　array pointer
    # 값복사없이쓸수있음
    for i in range(lab_array.shape[0]):
        inds = np.argwhere(~np.isnan(lab_array[i,:]))
        if inds.size == 0:
            lab_array[i,:] = get_avg_labtest(i)
        elif inds.size == 1:
            lab_array[i,:] = lab_array[i,inds[0,0]]
        else:
            prev_ind = inds[:,0][0]
            for ind in inds[:,0][1:]:
                prev_value = lab_array[i,prev_ind]
                curr_value = lab_array[i,ind]
                for input_index in range(prev_ind,ind+1):
                    lab_array[i,input_index] =\
                    (curr_value-prev_value)/(ind-prev_ind)*(input_index-prev_ind)+prev_value 
                prev_ind = ind
            lab_array[i,:inds[:,0][0]]=lab_array[i,inds[:,0][0]]
            lab_array[i,inds[:,0][-1]:]=lab_array[i,inds[:,0][-1]]
    return lab_df

@memoize
def get_avg_labtest(idx):
    global LABTEST_PATH
    lab_store = pd.HDFStore(LABTEST_PATH,mode='r')
    try:
        mapping_table = lab_store.select('metadata/mapping_table')
    finally:
        lab_store.close()
    avg_value = mapping_table.iloc[idx]['AVG']
    min_value = mapping_table.iloc[idx]['MIN']
    max_value = mapping_table.iloc[idx]['MAX']
    return (avg_value-min_value)/(max_value-min_value)

def get_patient_ts_df(no,t_day,f_day):
    demo_df = demo.get_timeserial_demographic(no).loc[:,f_day:t_day]
    pres_df = pres.get_timeserial_prescribe_df(no).loc[:,f_day:t_day]    
    lab_df = lab.get_timeserial_lab_df(no).loc[:,f_day:t_day]
    lab_df = impute_lab_basic(lab_df)
    diag_df = diag.get_timeserial_diagnosis_df(no).loc[:,f_day:t_day]
    # 기간동안 ２번이상 진단코드가 있으면，그 사이를 채움
    code_count = diag_df.sum(1)
    for code, _ in code_count[code_count >1].items():
        ts_series = diag_df.loc[code,:]
        first_code_day = ts_series[ts_series==1].index[0]
        last_code_day = ts_series[ts_series==1].index[-1]
        diag_df.loc[code,first_code_day:last_code_day] = 1.0
            
    return pd.concat([demo_df,pres_df,diag_df,lab_df])

def _make_patient_dataset(skip_df):
    global GAP_PERIOD, TARGET_PERIOD
    result = []
    for _, row in skip_df.iterrows():
        t_day = row.date - np.timedelta64(GAP_PERIOD,'D')
        f_day = t_day - np.timedelta64(TARGET_PERIOD-1,'D')

        df = get_patient_ts_df(row.no,t_day,f_day)\
                .fillna(0.0)\
                .as_matrix()
        result.append(df)
    return np.stack(result), skip_df.label.values

def make_patient_dataset(lab_test, diag_counts=None, pres_counts=None, lab_counts=None,types='train'):
    skip_df = skip_case(lab_test, diag_counts, pres_counts, lab_counts,types)

    min_label_size = min(skip_df[skip_df.label==0].shape[0],
    skip_df[skip_df.label==1].shape[0],
    skip_df[skip_df.label==2].shape[0])

    label_0_df = skip_df[skip_df.label==0].sample(min_label_size)
    label_1_df = skip_df[skip_df.label==1].sample(min_label_size)
    label_2_df = skip_df[skip_df.label==2].sample(min_label_size)

    concat_df = pd.concat([label_0_df,label_1_df,label_2_df])
    concat_df = concat_df.sample(3*min_label_size)

    pool = Pool()

    result = pool.map_async(_make_patient_dataset,np.array_split(concat_df,12))
    dataset_x = np.concatenate([x for x,_ in result.get()])
    dataset_y = np.concatenate([y for _,y in result.get()])

    return np.stack([dataset_x],axis=3), dataset_y

def get_patient_dataset_size(lab_test, diag_counts=None, pres_counts=None, lab_counts=None,types='train'):
    skip_df = skip_case(lab_test, diag_counts, pres_counts, lab_counts,types)
    min_label_size = min(skip_df[skip_df.label==0].shape[0],
    skip_df[skip_df.label==1].shape[0],
    skip_df[skip_df.label==2].shape[0])

    return min_label_size*3

