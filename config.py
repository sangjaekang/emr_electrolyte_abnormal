# -*- coding: utf-8 -*-
import sys, os, re
import pandas as pd
import numpy as np
#import tensorflow as tf
os_path = os.path.abspath('./')
find_path = re.compile('emr_electrolyte_abnormal')
BASE_PATH = os_path[:find_path.search(os_path).span()[1]]
sys.path.append(BASE_PATH)

# DATA PATH
# RAW data의 경로 
RAW_DATA_DIR = BASE_PATH + "/data/rawdata/potassium_data/"
RAW_LABTEST_PATH = RAW_DATA_DIR+"labtest.dat"
RAW_DIAGNOSIS_PATH = RAW_DATA_DIR+"diagnosis.dat"
RAW_PRESCRIBE_PATH = RAW_DATA_DIR+'prescribe.dat'
RAW_DEMO_PATH = RAW_DATA_DIR + 'medi_info.xlsx'

KCD_PATH = RAW_DATA_DIR + 'KCD.xlsx'
MEDICINE_CONTEXT_PATH = RAW_DATA_DIR + 'medicine_context.xlsx'

# 전처리된 data의 경로
PREP_DIR = BASE_PATH + '/data/prep/' 
LABTEST_PATH = PREP_DIR + 'labtest.h5'
DIAGNOSIS_PATH = PREP_DIR + 'diagnosis.h5'
PRESCRIBE_PATH = PREP_DIR + 'prescribe.h5'
LABEL_PATH = PREP_DIR + 'label.h5'
DEMO_PATH = PREP_DIR + 'demo.h5'

FEATURE_DIAGNOSIS_PATH = PREP_DIR + 'feature_selection_diagnosis.h5'
FEATURE_PRESCRIBE_PATH = PREP_DIR + 'feature_selection_prescribe.h5'

# RAW DATA 특성 
DELIM = '\x0b'  # 구분자

RAW_LABTEST_COLS = ['no','lab_test','date','result'] # RAW lab test data의 column 순서
RAW_DIAGNOSIS_COLS = ['no','date','KCD_code','description'] # Raw Diagnosis data의 column 순서
MEDICINE_CONTEXT_COLS = ['medi_code','medi_name','s_date','e_date','ingd','ATC_code','ATC_desc'] # medicine 정보 dataframe의 column순서
RAW_PRESCRIBE_COLS = ['no','medi_code','ingd_name','date','total','once','times','day'] # 
RAW_DEMO_COLS = ['no','sex','age']

MIN_DATE = '20110601' # 데이터 시작 날짜
MAX_DATE = '20170630' # 데이터 종료 날짜

# fdjLABTEST의 응급코드와 비응급코드 간 mapping  dictionary
EMGCY_AND_NOT_DICT = {
    'L8031':'L3011',# 총단백
    'L8032':'L3012',# 알부민
    'L8049':'L3013',# 콜레스테롤
    'L8036':'L3018',# 당정량
    'L8037':'L3019',# 요소질소
    'L8038':'L3020',# 크레아티닌
    'L8041':'L3041',# 소디움
    'L8042':'L3042',# 포타슘
    'L8043':'L3043',# 염소
    'L8044':'L3044',# 혈액총탄산
    'L8046':'L3022',# 총칼슘
    'L8047':'L3023',# 인
    'L8048':'L3021',# 요산
    'L8049':'L3013',# 콜레스테롤
    'L8050':'L3029',# 중성지방
    'L8053':'L3057' # LDH
}

# 데이터 전처리 내 에서의 configuration
CORE_NUMS = 12 # cpu core 수

SKIP_TEST_INTERVAL = 7 # LAB 데이터 내 검사 일자 간 최소간격
SKIP_LABEL_INTERVAL = 30 # 전해질 이상 발병후 최소간격 (이거보다 길어야지 독립적인 발병)

# Input data의 최소 데이터 갯수
# 이것보다 적으면 too much sparse로 어려움
SKIP_DIAG_COUNTS = 1 # INPUT DATA 중 최소 진단 코드의 횟수
SKIP_LAB_COUNTS = 10 # INPUT DATA 중 최소 lab test의 횟수

KCD_MAP_TYPE = 1 # diagnosis 내 KCD code를 변환시킬 때 기준 
                                  # [0:세분류명 1:소분류명 2:중분류명 ３:대분류명]

KCD_COUNT_STANDARD = 1000 # diagnosis data 내 KCD code의 진단 최소 건수 기준
                                                       # 너무 적은 케이스를 지워서，희소한 KCD code를 삭제 

MEDICINE_COUNT_STANDARD = 1000 # prescribe data 내 약 code의 처방 최소 건수 기준
                                                                # 너무 적은 케이스를 지워서，희소한 약 code를 삭제 
# DEBUG 
DEBUG_PRINT = True

# 데이터예측시 시간간격
#   시간순 －－－－－－＞
#        TARGET_PERIOD <-> GAP_PERIOD <-> PREDICTION_PERIOD
GAP_PERIOD = 7 # TARGET_PERIOD와 PREDICITON_PERIOD 사이 gap 기간
TARGET_PERIOD = 180 # prediction 할 때 Input 시간
PREDICTION_PERIOD = 60 # prediction 할 때 예측범위 시간