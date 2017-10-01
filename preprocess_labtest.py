# -*- coding: utf-8 -*-
import sys, os, re

os_path = os.path.abspath('./') ; find_path = re.compile('emr_electrolyte_abnormal')
BASE_PATH = os_path[:find_path.search(os_path).span()[1]]
sys.path.append(BASE_PATH)

from config import *
'''
LABTEST DATA I/O관련　모듈
전처리하고，　환자　관련된　time-serial dataframe을　출력하는　함수

핵심　메소드
    * get_timeserial_lab_df(no)
        환자별　timeserial한　lab_test 결과를　출력하는　메소드
        ----
        output matrix property
        
            row : lab_test name (labtest 종류)
            columns : time-serial column (MIN DATE 부터　MAX DATE까지　time-serial datetime) 
            value : lab_test result (환자의　labtest 결과)
    
    * preprocess_labtest()
        raw data ( 병원에서　받은　original 자료)를　hdf5　포맷에　맞게　정규화시켜　저장하는　메소드
        ----
        output hdf5 property

            original : raw data 중　필요없는　환자를　제거만　하고　저장한 dataframe
            prep : raw data를　정규화하여　저장한　dataframe
            metadata/usecol : 각　labtest　별　case 갯수
            metadata/mapping_table : 정규화에　사용된　mapping_table
'''
def get_timeserial_lab_df(no):
    # 환자에 대한 시계열 'lab_test' dataframe을 구하는 함수
    global DEBUG_PRINT, MIN_DATE, MAX_DATE, LABTEST_PATH
    
    #전처리된　데이터가　없으면　전처리하여　생성
    if not os.path.isfile(LABTEST_PATH):
        if DEBUG_PRINT: print("no LABTEST file")
        preprocess_labtest()

    lab_store = pd.HDFStore(LABTEST_PATH,mode='r')
    try:
        target_df = lab_store.select('prep',where='no=={}'.format(no))
        usecol = lab_store.select('metadata/usecol')
    finally:
        lab_store.close()
    _y = target_df[['lab_test','date','result']].\
           pivot_table(index=['lab_test'],columns=['date'])
    _y.columns= _y.columns.droplevel()

    return _y.reindex(index=usecol.index,columns=pd.date_range(MIN_DATE,MAX_DATE,freq='D'))

def preprocess_labtest():
    # RAW labtest data를 전처리하는 함수 
    global DEBUG_PRINT, RAW_LABTEST_PATH, LABTEST_PATH, DELIM, RAW_LABTEST_COLS, SKIP_TEST_INTERVAL
    
    if DEBUG_PRINT: print("preprocess_labtest starts")
    
    if not os.path.isfile(RAW_LABTEST_PATH):
        raise ValueError(' there is no RAW_LABTEST file (wrong path)')

    lab_df = pd.read_csv(RAW_LABTEST_PATH,delimiter=DELIM,header=None,names=RAW_LABTEST_COLS)
    # 이전 전처리된 파일이 있으면 삭제
    if os.path.isfile(LABTEST_PATH):
        os.remove(LABTEST_PATH)

    # date의 포맷을 DATETIME FORMAT으로 변경
    lab_df.loc[:,'date'] = pd.to_datetime(lab_df['date'].astype(str), format='%Y%m%d')
    
    no_lab_times = lab_df[['no','date']].drop_duplicates()
    #검사를 한번만（하루만） 받은 사람들 제거
    no_once = no_lab_times.groupby('no').count()[no_lab_times.groupby('no').count() > 1].date
    no_lab = no_once[~np.isnan(no_once)].index

    #환자가 받은 검사 중에서 제일 마지막에 받은 검사 날짜와 제일 처음 받은 검사 날짜를 뺌
    test_interval = no_lab_times[no_lab_times.no.isin(no_lab)].groupby('no').date.max()-\
                    no_lab_times[no_lab_times.no.isin(no_lab)].groupby('no').date.min()
    
    # 뺀 날짜가 SKIP_PERIOD보다 작은 경우，그 경우는 무시
    # 이 경우를 빼지 않으면，짧은 입원（２～３일）에 여러번 받은 경우의 환자가 들어가게 됨
    no_exist = test_interval[test_interval>np.timedelta64(SKIP_TEST_INTERVAL,'D')].index
    pre_lab_df = lab_df[lab_df.no.isin(no_exist)]
    # 응급코드를 비응급코드로 변환
    pre_lab_df.loc[:,'lab_test']=pre_lab_df.lab_test.map(replace_emgcy_code)
    # 각 코드별로 min~max가 다르므로, 각 코드별 정규화을 위한 mapping table 생성
    mapping_table = set_mapping_table(pre_lab_df)

    # original value로 저장
    orginal_lab_df = pre_lab_df.copy()
    orginal_lab_df.loc[:,'result'] = orginal_lab_df.lab_test.map(change_number)
    orginal_lab_df.to_hdf(LABTEST_PATH,'original',format='table',data_columns=True,mode='a')

    # 각 검사별로 normalizing 적용
    lab_list = []
    for lab_name in pre_lab_df.lab_test.unique():
        _lab_df = pre_lab_df.loc[pre_lab_df.lab_test==lab_name]
        r_avg=mapping_table['AVG'][lab_name]
        r_min=mapping_table['MIN'][lab_name]
        r_max=mapping_table['MAX'][lab_name]
        _lab_df.loc[:,'result'] = _lab_df.result.map(normalize_number(r_avg,r_min,r_max))
        if DEBUG_PRINT: print("    normalize {} ends".format(lab_name))
        lab_list.append(_lab_df)
    
    normalized_lab_df = pd.concat(lab_list)        
    normalized_lab_df.dropna(inplace=True)
    normalized_lab_df.to_hdf(LABTEST_PATH,'prep',format='table',data_columns=True,mode='a')
    
    # count_lab_df : lab_test 별로 몇건이 있는지 저장
    count_lab_df = normalized_lab_df[['no','lab_test']].groupby('lab_test').count()
    count_lab_df.columns = ['counts']
    count_lab_df.to_hdf(LABTEST_PATH,'metadata/usecol',format='table',data_columns=True,mode='a')

    if DEBUG_PRINT: print("preprocess_labtest ends")

def set_mapping_table(lab_df):
    '''
    labtest의 mapping table을 생성하는 함수
    평균/ 최솟값/최댓값으로 구성
    이를 hdf5파일의 metadata에 저장
    '''
    global DEBUG_PRINT, LABTEST_PATH
    if DEBUG_PRINT: print("set_mapping_table starts")
    
    result_df = pd.DataFrame(columns=['lab_test','AVG','MIN','MAX'])
    for lab_name in lab_df.lab_test.unique():
        per_lab_df = lab_df.loc[lab_df.lab_test == lab_name,['no','result']]
        # 1. 숫자로 치환하기
        per_lab_df.result = per_lab_df.result.map(change_number)
        # 2. 이상 값 처리 시 대응되는 값
        r_avg   = revise_avg(per_lab_df.result)
        r_min  = revise_min(per_lab_df.result)
        r_max = revise_max(per_lab_df.result)
        # 3. save
        result_df = result_df.append({'lab_test':lab_name,'AVG':r_avg,'MIN':r_min,'MAX':r_max}, ignore_index=True)        
        if DEBUG_PRINT: print("     write {} completed".format(lab_name))

    result_df=result_df.set_index('lab_test')
    result_df.to_hdf(LABTEST_PATH,'metadata/mapping_table',format='table',data_columns=True,mode='a')
    if DEBUG_PRINT: print("set_mapping_table ends")
    return result_df.to_dict()


def change_number(x):    
    # 숫자 표현을 통일  (범위 쉼표 등 표현을 단일표현으로 통일）
    str_x = str(x).replace(" ","")
    re_num   = re.compile('^[+-]{0,1}[\d\s]+[.]{0,1}[\d\s]*$') #숫자로 구성된 데이터를 float로 바꾸어 줌
    re_comma = re.compile('^[\d\s]*,[\d\s]*[.]{0,1}[\d\s]*$') # 쉼표(,)가 있는 숫자를 선별
    re_range = re.compile('^[\d\s]*[~\-][\d\s]*$') # 범위(~,-)가 있는 숫자를 선별
    if re_num.match(str_x):
        return float(str_x)
    else:
        if re_comma.match(str_x):
            return change_number(str_x.replace(',',""))
        elif re_range.match(str_x):
            if "~" in str_x:
                a,b = str_x.split("~")
            else:
                a,b = str_x.split("-")
            return np.mean((change_number(a),change_number(b)))
        else :
            return np.nan

def revise_avg(x):
    # 10~90% 내에 있는 값을 이용해서 평균 계산
    quan_min = x.quantile(0.10)
    quan_max = x.quantile(0.90)
    return x[(x>quan_min) & (x<quan_max)].mean()

def revise_std(x):
    # 1~99% 내에 있는 값을 이용해서 표준편차 계산
    quan_min = x.quantile(0.01)
    quan_max = x.quantile(0.99)
    return x[(x>quan_min) & (x<quan_max)].std()

def revise_min(x):
    # 3시그마 바깥 값과 quanter 값의 사이값으로 결정
    std_min = revise_avg(x)-revise_std(x)*3 # 3 시그마 바깥 값
    q_min = x.quantile(0.01)
    if std_min<0 :
        # 측정값중에서 음수가 없기 때문에, 음수인 경우는 고려안함
        return q_min
    else :
        return np.mean((std_min,q_min))

def revise_max(x):
    # 3시그마 바깥 값과 quanter 값의 사이값으로 결정
    std_max = revise_avg(x)+revise_std(x)*3
    q_max = x.quantile(0.99)
    return np.mean((std_max,q_max))


def replace_emgcy_code(x):
    # 응급코드를 비응급코드로 바꾸어주는 함수
    global EMGCY_AND_NOT_DICT
    if x in EMGCY_AND_NOT_DICT:
        return EMGCY_AND_NOT_DICT[x]
    else:
        return x

def normalize_number(mean_x,min_x,max_x):
    '''
    dataframe 내 이상값을 전처리하는 함수.
    dataframe.map 을 이용할 것이므로, 함수 in 함수 구조 사용
    '''
    def _normalize_number(x):
        str_x = str(x).strip()

        re_num = re.compile('^[+-]?[\d]+[.]?[\d]*$')
        re_lower = re.compile('^<[\d\s]*[.]{0,1}[\d\s]*$')
        re_upper = re.compile('^>[\d\s]*[.]{0,1}[\d\s]*$')
        re_star = re.compile('^[\s]*[*][\s]*$')
        if re_num.match(str_x):
            # 숫자형태일경우
            float_x = np.float(str_x)
            if float_x > max_x: return 1
            elif float_x < min_x: return 0
            else: return (np.float(str_x) - min_x)/(max_x-min_x)
        else:
            if re_lower.match(str_x): return 0
            elif re_upper.match(str_x): return  np.float(1)
            elif re_star.match(str_x): return np.float( (mean_x-min_x)/(max_x-min_x) )
            else: return np.nan
    return _normalize_number

