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
DEMOGRAPHIC DATA I/O 관련 모듈
demographic data를 전처리하고， 환자 관련된 time-serial dataframe을 출력하는 함수

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


def get_timeserial_demographic(no):
    global DEMO_PATH
    #전처리된 데이터가 없으면 전처리하여 생성
    if not os.path.isfile(DEMO_PATH):
        if DEBUG_PRINT:
            print("no demographic file")
        preprocess_demographic()

    demo_store = pd.HDFStore(DEMO_PATH, mode='r')
    try:
        demo_df = demo_store.select('prep', where='no=={}'.format(no))
    finally:
        demo_store.close()
    min_year = int(MIN_DATE) // 10000
    max_year = int(MAX_DATE) // 10000

    sex = demo_df.sex.values[0]
    age = demo_df.age.values[0]
    result_df = pd.DataFrame(index=['sex','age'], columns=pd.date_range(MIN_DATE, MAX_DATE, freq='D'))
    for year in range(max_year, min_year-1, -1):
        result_df.loc['sex', result_df.columns[result_df.columns.year == year]] = sex
        result_df.loc['age', result_df.columns[result_df.columns.year == year]] = age
        age = age-0.01
    return result_df


def preprocess_demographic():
    global RAW_DEMO_COLS, DEMO_PATH, RAW_DEMO_PATH, DEBUG_PRINT, MIN_DATE, MAX_DATE
    if DEBUG_PRINT:
        print("preprocess_demo starts")
    demo_df = pd.read_excel(RAW_DEMO_PATH)
    demo_df.columns = RAW_DEMO_COLS
    min_year = int(MIN_DATE) // 10000
    max_year = int(MAX_DATE) // 10000
    #나이를　１~100세　범위로　바꾸어줌
    # max_year-min_year한것은　
    # max_year에서　나이가　６살이면　min_year에서　나이가　１살로　되기　위함
    demo_df.loc[:, 'age'] = np.clip(demo_df.age, max_year-min_year, 100)/100
    demo_df.loc[:, 'age'] = demo_df.loc[:, 'age']/100
    # 0~1사이　값으로　정규화
    # Female : 1 Male : 0 으로　치환
    demo_df.loc[:,'sex'] = demo_df.sex.map(lambda x: 1 if x == 'F' else 0)
    demo_df.to_hdf(DEMO_PATH, 'prep', format='table', data_columns=True, mode='a')
    if DEBUG_PRINT:
        print("preprocess_demo ends")