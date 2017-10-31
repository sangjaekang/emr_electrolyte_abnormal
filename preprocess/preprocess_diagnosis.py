# -*- coding: utf-8 -*-
import sys
import os
import re

os_path = os.path.abspath('./')
find_path = re.compile('emr_electrolyte_abnormal')
BASE_PATH = os_path[:find_path.search(os_path).span()[1]]
sys.path.append(BASE_PATH)

from config import *
'''
DIAGNOSIS DATA I/O 관련 모듈
diagnosis data를 전처리하고， 환자 관련된 time-serial dataframe을 출력하는 함수

핵심 메소드
    * get_timeserial_diagnosis_df(no)
        환자별 time-serial한 diagnosis 결과를 출력하는 메소드
        ----
        output matrix property
            
            row : KCD_code
            columns  : time-serial column (MIN DATE 부터 MAX DATE까지 time-serial datetime) 
            value :   1-발병O / 0-발병X
    * preprocess_diagnosis()
        raw data ( 병원에서 받은 original 자료)를 전처리하여 저장하는 메소드
        ----
        output hdf5 property
            prep : raw data중 KCD_Code를 통일시키고， 불필요 code를 제거하여 저장한 dataframe
            metadata/usecol : 각 KCD_code 별 case 갯수
'''


def get_timeserial_diagnosis_df(no, feature_selected=True):
    # 환자에 대한 시계열  진단 dataframe을 구하는 함수
    global DEBUG_PRINT, MIN_DATE, MAX_DATE, DIAGNOSIS_PATH

    # 전처리된 데이터가 없으면 전처리하여 생성
    if not os.path.isfile(DIAGNOSIS_PATH):
        if DEBUG_PRINT:
            print("no DIAGNOSIS file")
        preprocess_diagnosis()

    diagnosis_store = pd.HDFStore(DIAGNOSIS_PATH, mode='r')
    try:
        target_df = diagnosis_store.select('prep', where='no=={}'.format(no))
        if feature_selected:
            usecol = diagnosis_store.select('metadata/boruta').code.values
        else:
            usecol = diagnosis_store.select('metadata/usecol').index
    finally:
        diagnosis_store.close()
    _y = target_df[['no', 'date', 'KCD_code']]\
        .pivot_table(index=['KCD_code'], columns=['date'])\
        .applymap(lambda x: 1.0 if not np.isnan(x) else np.nan)

    _y.columns = _y.columns.droplevel()
    return _y.reindex(index=usecol, columns=pd.date_range(MIN_DATE, MAX_DATE, freq='D')).fillna(axis=1,method='ffill').fillna(0)


def preprocess_diagnosis():
    # RAW diagnosis data를 전처리하는 함수
    global DEBUG_PRINT, RAW_DIAGNOSIS_PATH, DIAGNOSIS_PATH, DELIM, RAW_DIAGNOSIS_COLS, KCD_MAP_TYPE, KCD_COUNT_STANDARD
    if DEBUG_PRINT:
        print("preprocess_diagnosis starts")
    diagnosis_df = pd.read_csv(RAW_DIAGNOSIS_PATH, delimiter=DELIM, header=None, usecols=[
                               0, 1, 2, 3], names=RAW_DIAGNOSIS_COLS)

    # 전처리전　저장
    diagnosis_df.to_hdf(DIAGNOSIS_PATH, 'original',
                        format='table', data_columns=True, mode='a')
    # KCD 코드 전처리
    # KCD 코드를 분류단위（세분류，소분류，중분류，대분류）중 하나를 기준으로 통일
    diagnosis_df.loc[:, 'KCD_code'] = diagnosis_df.KCD_code.map(
        map_KCD_by_type(KCD_MAP_TYPE))
    # KCD 코드 중 발병이 적은 경우들을（KCD_COUNT_STANDARD 보다 적은 경우）지움
    if DEBUG_PRINT:
        print("    KCD Mapping code 건수 최소 기준 : {}".format(KCD_COUNT_STANDARD))
    pre_diagnosis_df = diagnosis_df.groupby('KCD_code').filter(
        lambda x: len(x) > KCD_COUNT_STANDARD)
    # date의 포맷을 DATETIME FORMAT으로 변경
    pre_diagnosis_df.loc[:, 'date'] = pd.to_datetime(
        pre_diagnosis_df['date'].astype(str), format='%Y%m%d')

    pre_diagnosis_df.to_hdf(DIAGNOSIS_PATH, 'prep',
                            format='table', data_columns=True, mode='a')
    # KCD_count_df : KCD_code 별로 몇 건이 있는지 저장
    KCD_count_df = pre_diagnosis_df[
        ['KCD_code', 'date']].groupby('KCD_code').count()
    KCD_count_df.columns = ['counts']
    KCD_count_df.to_hdf(DIAGNOSIS_PATH, 'metadata/usecol',
                        format='table', data_columns=True, mode='a')

    if DEBUG_PRINT:
        print("preprocess_diagnosis ends")


def map_KCD_by_type(kind=0):
    global KCD_PATH
    KCD_df = pd.read_excel(KCD_PATH)
    if kind == 0:
        kind = '세분류명'
    elif kind == 1:
        kind = '소분류명'
    elif kind == 2:
        kind = '중분류명'
    elif kind == 3:
        kind = '대분류명'
    else:
        raise ValueError(
            "there is none type, type need to be integer in range from 0 to 3 ")

    if DEBUG_PRINT:
        print("    KCD Mapping code 기준 : {}".format(kind))
    KCD_series = pd.Series(
        index=KCD_df['진단용어코드'].values, data=KCD_df[kind].values)
    KCD_map_dict = KCD_series.to_dict()
    return KCD_map_dict
