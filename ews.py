"""<데이터 기반 금융˙외환 조기경보모형*을 위한 라이브러리>
* BOK 이슈노트 No.2024-11(김태완, 박정희, 이현창, 2024)

조기경보모형 관련 데이터 입수/변환, 하이퍼파라미터 튜닝 및 평가, 학습 및 예측을 위해 필요한 클래스와 함수로 구성

주요 클래스:
- Bidas: Bidas 데이터를 API나 엑셀 파일로부터 입수하기 위한 클래스
- SignalExtraction: (Scikit-learn 호환) 신호추출법 모형의 구현체
- EarlyWarningModel: 조기경보모형의 학습, 실행, 저장, 로딩을 위한 클래스

주요 함수:
- preprocess: CFPI 구성변수 및 조기경보모형 입력변수를 산정
- get_crises: CFPI가 임계치를 넘는 기간을 기준으로, 위기기간 및 학습데이터의 그룹을 산정
- run_cv: 각 모델에 대해 하이퍼파라미터 튜닝한 후 예측결과를 산출
"""
import pickle
import warnings
import os

from arch import arch_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.pipeline import Pipeline
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from tqdm.auto import tqdm
from pandas.tseries.offsets import DateOffset
from dateutil.relativedelta import relativedelta


# 기본 설정
warnings.filterwarnings(action='ignore')
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# 기본 팔레트값
DEFAULT_PALETTE = {
    '자금조달': 'olive',
    '레버리지': 'tan',
    '자산가격': 'orange',
    '금리': 'darkgreen',
    '변동성': 'royalblue',
    '경기': 'grey',
    '대외': 'black',
    '심리': 'navy'
}

def set_all_seeds(seed=123):
    """모든 랜덤 시드 설정

    Args:
        seed: 설정할 시드값
    """
    import random
    np.random.seed(seed)  # numpy 랜덤 시드
    random.seed(seed)     # Python 기본 랜덤 시드
    # scikit-learn의 일부 함수들도 random을 사용하므로, 필요시 추가 설정 가능
    # 예: sklearn.utils.check_random_state(seed)
    
def load_params(file_path, model_name='ET'):
    params_df = pd.read_pickle(file_path)
    params = params_df[model_name]
    params_dict = params.dropna().to_dict()
    return {k: int(v) if isinstance(v, float) and v.is_integer() else v for k, v in params_dict.items()}

# I. 데이터 입수 및 변환

class Bidas:
    """Bidas 시계열 데이터를 API나 엑셀 파일로부터 입수하기 위한 클래스"""

    def __init__(self, source_type='API', api_headers={}, file_name='', file_sheet='data'):
        assert source_type in ['API', 'Excel', 'GoogleDocs'] # assert: 조건이 참이 아니면 에러 발생, 소스 유형 타입 확인
        self.source_type = source_type
        self.api_headers = api_headers # 필요시 사용자 인증 정보 포함
        self.file_name = file_name # 로컬 엑셀파일인 경우 파일명, 구글닥스인 경우 파일id
        self.file_sheet = file_sheet # 엑셀 시트명
        self.series = {} # Bidas id별 시계열 데이터
        self.freqs = {} # Bidas id별 시계열 데이터 빈도

    def load_series(self, bidas_ids):
        """데이터 소스유형에 따라 Bidas 시계열 데이터를 읽어온다."""
        if self.source_type == 'API':
            self._load_series_from_api(bidas_ids)
        else:
            self._load_series_from_file(bidas_ids)

    def _load_series_from_api(self, bidas_ids):
        """Bidas API를 이용하여 Bidas 시계열 데이터를 읽어온다."""
        api_url = 'http://datahub.boknet.intra/api/v1/obs/lists'
        res = requests.post(api_url, headers=self.api_headers, data={'ids': bidas_ids})
        for raw_data in res.json()['data'][0]:
            try:
                bidas_id = raw_data['series_id']
                obs = pd.DataFrame(raw_data['observations'])
                series = obs['value'].apply(lambda x: np.nan if x == '' else x).set_axis(pd.to_datetime(obs['period'])).dropna()
                series.name = bidas_id
                if series.dtype == 'O':
                    series = series.str.strip().str.replace(',', '').astype(float)
                self.series[bidas_id] = series
                self.freqs[bidas_id] = self._get_freq(series)
            except:
                print(f'{bidas_id} is not loaded.')

    def _load_series_from_file(self, bidas_ids):
        """Bidas 엑셀 플러그인을 통해 미리 데이터를 다운받은 엑셀 파일로부터 Bidas 시계열 데이터를 읽어온다."""
        if self.source_type == 'Excel': # 로컬 엑셀 파일
            # 엑셀에서는 Bidas id(1행)과 메타데이터(2~13행)를 각각의 행으로 인식 → 메타데이터 행 skip
            raw_data = pd.read_excel(self.file_name, header=0, skiprows=range(1, 13), sheet_name=self.file_sheet)
        else: # 구글닥스로 변환한 엑셀 파일
            gd_url = 'https://docs.google.com/spreadsheets/d/{0}/gviz/tq?tqx=out:csv&sheet={1}'
            file_url = gd_url.format(self.file_name, self.file_sheet)
            raw_data = pd.read_csv(file_url, header=0, low_memory=False)
            # 구글닥스에서는 Bidas id와 메타데이터를 합쳐서 하나의 행으로 인식 → Bidas id를 분리하여 홀수번째열에 배정
            raw_data.columns = [f'Unnamed: {i}' if i % 2 else column.split(' ')[0] for i, column in enumerate(raw_data.columns)]
        for bidas_id in bidas_ids:
            try:
                # i열에는 Bidas id 및 period가, i+1열에는 각 period별 데이터값이 존재
                index = raw_data.columns.get_loc(bidas_id)
                series = raw_data.iloc[0:, index+1].set_axis(pd.to_datetime(raw_data.iloc[0:, index])).rename_axis('period').dropna()
                series.name = bidas_id
                if series.dtype == 'O':
                    series = series.str.strip().str.replace(',', '').astype(float)
                self.series[bidas_id] = series
                self.freqs[bidas_id] = self._get_freq(series)
            except:
                print(f'{bidas_id} is not loaded.')

    def _get_freq(self, series):
        """입력받은 시계열 데이터의 빈도를 데이터포인트간의 시점차이로 추정한다.

        Args:
            series: 빈도를 추정할 시계열

        Returns:
            freq: 추정된 빈도(D:일별, M:월별, Q:분기별, A:연별)
        """
        freq = None
        interval = pd.to_timedelta(np.diff(series.index).min()).days # np.diff(series.index).min(): 시계열 데이터의 인덱스 간격 중 최소값, pd.to_timedelta(): 인덱스 간격을 시간 간격으로 변환, .days: 시간 간격을 일수로 변환
        if interval == 1:
            freq = 'D'
        elif 28 <= interval <= 31:
            freq = 'M'
        elif 90 <= interval <= 92:
            freq = 'Q'
        elif 365 <= interval <= 366:
            freq = 'A'
        return freq

    def get_table(self, bidas_ids, freq, range_from=None, range_to=None, downsample='mean', upsample='ffill'):
        """Bidas 시계열 데이터를 지정한 빈도에 맞게 변환하여 제공한다.

        Args:
            bidas_ids: Bidas 시계열 아이디 목록
            freq: 출력 시계열의 빈도 (D:일별, M:월별, Q:분기별, A:연별)
            range_from: 데이터 시작일자(YYYY-MM-DD)
            range_to: 데이터 종료일자(YYYY-MM-DD)
            downsample: {freq}보다 원본 데이터의 빈도가 높을 경우 다운샘플링 방법(e.g. mean, sum, max)
            upsample: {freq}보다 원본 데이터의 빈도가 낮을 경우 업샘플링 방법(e.g. ffill, bfill)

        Returns:
            table: 빈도가 {freq}로 일치된 시계열 데이터 테이블
        """
        new_bidas_ids = [bidas_id for bidas_id in bidas_ids if bidas_id not in self.series]
        if len(new_bidas_ids) > 0:
            self.load_series(new_bidas_ids)
        table = pd.DataFrame()
        intervals = {'D': 1, 'M': 30, 'Q': 90, 'A': 365}
        table_interval = intervals[freq]
        for bidas_id in bidas_ids:
            series_interval = intervals[self.freqs[bidas_id]]
            if series_interval < table_interval:
                ts = self.series[bidas_id][range_from:range_to].resample(freq).agg(downsample).to_period(freq)
            else:
                ts = self.series[bidas_id][range_from:range_to].resample(freq).agg(upsample).to_period(freq)
            table = table.join(ts, how='outer')
        return table


class Transform:
    """시계열 데이터 변환을 위한 utility 클래스"""

    @staticmethod
    def scale(x):
        """샘플의 평균과 표준편차로 표준화"""
        result = (x - x.mean())/x.std()
        return result

    @staticmethod
    def link(x):
        """단절된 시계열을 앞선 시계열의 가중치를 조정하여 연결"""
        prev_weight = 1
        prev_result = x.iloc[:, 0]
        for i in range(len(x.columns)-1):
            link = x.iloc[:, i].dropna().index.min()
            weight = prev_weight = prev_weight * x.iloc[:, i].loc[link] / x.iloc[:, i+1].loc[link]
            result = prev_result = pd.concat([x.iloc[:, i+1].loc[:link].iloc[:-1] * weight, prev_result.loc[link:]])
        return result.to_frame()

    @staticmethod
    def cmax(ts):
        """CMAX 계산"""
        result = (-1*ts/ts.rolling(24).max())
        return result

    @staticmethod
    def beta(x):
        """beta 계산"""
        ret = pd.DataFrame({'ts': x.iloc[:, 0], 'base': x.iloc[:, 1]}).pct_change(12) * 100
        cov = ret.rolling(12).cov().unstack()['base']['ts']
        var = ret.rolling(12).var()['base']
        ret['beta'] = cov.div(var, axis=0)
        result = ret.apply(lambda x: x.beta if x.beta >= 1 and x.ts < x.base else 0, axis=1).to_frame()
        return result

    @staticmethod
    def mvol(ts, horizon=6):
        """변동성(이동평균표준편차) 계산"""
        result = ts.pct_change().rolling(horizon).std()
        return result

    @staticmethod
    def gvol(ts, horizon=12, min_sample=30, recursive=False):
        """GARCH 변동성 계산"""
        min_sample = 30 if recursive else len(ts)
        scale = 100 if recursive else 1
        ts_diff = ts.pct_change().bfill() * scale
        for i in range(len(ts)-min_sample+1):
            garch_model = arch_model(ts_diff.iloc[:i+min_sample], mean='AR',
                                     lags=horizon, vol='Garch', p=1, q=1, rescale=False)
            result = garch_model.fit(update_freq=10, disp='off').conditional_volatility * 100
            results = result if i == 0 else pd.concat([results, result.iloc[-1:]])
        return results


def preprocess(data):
    """CFPI 구성변수 및 조기경보모형 입력변수를 산정한다.

    Args:
        data: 사전정의된 메타데이터에 따라 수집한 Bidas의 시계열 데이터 테이블

    Returns:
        data: Bidas 데이터를 바탕으로 새로이 산정한 변수를 포함한 데이터
    
    변수 변환기법:
        1. 차분/비율/부호변환
           - 차분: data['var1'] - data['var2']  (스프레드 계산)
           - 비율: data['var1'] / data['var2']  (레버리지, 예대율 등)
           - 부호 변환: -data['var']  (해석 방향 조정)

        2. 시계열 기본 변환
           - 차분: .diff(12)  (전년동기 대비 변화량)
           - 백분율 변화: .pct_change()  (성장률, 수익률)
           - 이동평균: .rolling(3).mean()  (평활화)
           - 이동합계: .rolling(12).sum()  (연간화)

        3. 복합 변환 (Transform 클래스)
           - CMAX: Transform.cmax()  (최고점 대비 하락률)
           - GARCH 변동성: Transform.gvol()  (조건부 변동성)
           - 이동 변동성: Transform.mvol()  (수익률 표준편차)
           - 시계열 연결: Transform.link()  (단절 시계열 연결)

        4. 경제지표 특화 변환
           - GDP 대비 비율: (신용/GDP.rolling(12).sum()).rolling(3).mean()
           - 이동평균 + 변화율: var.rolling(3).mean().pct_change()
           - 레벨 + 차분 조합: 원시값과 12개월 차분 동시 생성

        5. 데이터 결합
           - 열 합계: .dropna().sum(1)  (여러 시리즈 통합)
           - 시계열 연결: Transform.link()  (과거-현재 지수 연결)
    """
    # 0. 공통 변수(CFPI 구성 & 조기경보모형 입력변수)
    # CD 스프레드(CD수익률 - 통안증권 수익률)
    data['cd_sp'] = data['cd91'] - data['ms91']
    # CP 스프레드(CP91 - 콜금리)
    data['cp_sp'] = data['cp91'] - data['call']
    # (마이너스) 기간 프리미엄(국채3년물 - 통안증권1년물)
    data['tp_sp_neg'] = -(data['kb3y'] - data['ms1y'])
    # 은행업 지수(KOSPI 은행업지수 + KRX 은행업지수)
    data['post_kosbank'] = data['krxbank']['2020-06':] # KOSPI 만료(2022.06) 2년전을 시작점으로 설정
    data['stockbank'] = data[['post_kosbank', 'kosbank']].transform(Transform.link) # .transform(): 데이터프레임의 각 열에 함수 적용
    # 비은행업 지수I(KOSPI 증권업지수 + KRX 증권업지수)
    data['post_kossecu'] = data['krxsecu']['2008-01':] # KRX 시작(2006.01) 2년후를 시작점으로 설정
    data['stocksecu'] = data[['post_kossecu', 'kossecu']].transform(Transform.link)
    # 비은행업 지수II(KOSPI 보험업지수 + KRX 보험업지수)
    data['post_kosins'] = data['krxins']['2011-01':] # KRX 시작(2009.01) 2년후를 시작점으로 설정
    data['stockins'] = data[['post_kosins', 'kosins']].transform(Transform.link)

    # 1. CFPI 구성변수
    # 은행업 지수 변동성 (GARCH)
    data['bank_gv'] = data['stockbank'].transform(Transform.gvol)
    # 비은행업 지수 변동성I-1: 증권(GARCH)
    data['secu_gv1'] = data['kossecu'].transform(Transform.gvol)
    # 비은행업 지수 변동성II-1: 보험(GARCH)
    data['ins_gv1'] = data['kosins'].transform(Transform.gvol)
    # 비은행업 지수 변동성I-2: 증권(GARCH)
    data['secu_gv2'] = data['stocksecu'].transform(Transform.gvol)
    # 비은행업 지수 변동성II-2: 보험(GARCH)
    data['ins_gv2'] = data['stockins'].transform(Transform.gvol)
    # 회사채 스프레드(회사채AA - 국채3년)
    data['cr_sp'] = data['cbaa3y'] - data['kb3y']
    # (마이너스) 주식 수익률 (KOSPI 종가)
    data['stock_ret'] = -data['kospi'].pct_change(12)
    # 주식 변동성 (GARCH)
    data['stock_gv'] = data['kospi'].transform(Transform.gvol)
    # 미달러 환율 변동성 (GARCH)
    data['er_gv'] = data['er'].transform(Transform.gvol)

    # 2. 조기경보모형 입력변수
    ## 취약성/자금조달/대외
    # (마이너스) 단기외채 / 외환보유액 비율
    data['res_sdebt'] = data['reserve']/data['short_ex_debt']
    data['res_sdebt_neg'] = -data['res_sdebt']
    
    ## 취약성/자금조달/은행
    # 은행 레버리지 비율 12개월 차분
    data['bank_lev'] = data['bank_asset']/(data['bank_capital'])
    data['bank_lev_diff'] = data['bank_lev'].diff(12)
    # 은행 예대율 12개월 차분
    data['bank_ldr'] = (data['bank_loan'] / data['bank_dep'])*100
    data['bank_ldr_diff'] = data['bank_ldr'].diff(12)
    
    ## 취약성/자금조달/비은행
    # 저축은행 레버리지 비율 12개월 차분
    data['sbank_lev'] = data['sbank_asset']/(data['sbank_capital'])
    data['sbank_lev_diff'] = data['sbank_lev'].diff(12)
    # 비은행부문 여수신 비율
    data['nbank_ratio'] = data['nbank_rc']/(data['nbank_rc'])
    data['nbank_ratio_diff'] = data['nbank_ratio'].diff(12)
    
    ## 취약성/레버리지/가계
    # 가계신용 (hc1975 + hc2002 + hc2008)
    data['hc1975'] = data[['hc1975_lbond', 'hc1975_sbond', 'hc1975_loan', 'hc1975_gov']].dropna().sum(1)
    data['hc2002'] = data[['hc2002_bond', 'hc2002_loan', 'hc2002_gov']].dropna().sum(1)
    data['hc2008'] = data[['hc2008_bond', 'hc2008_loan', 'hc2008_gov']].dropna().sum(1)
    data['hc2'] = data[['hc2008', 'hc2002', 'hc1975']].transform(Transform.link)
    # GDP대비 가계신용 12개월 차분
    data['hc2_gdp'] = (data['hc2']/data['gdp'].rolling(12).sum()*3).rolling(3).mean()
    data['hc_gdp_diff'] = data['hc2_gdp'].diff(12)
    
    ## 취약성/레버리지/기업
    # 기업신용(cc1975 + cc2002 + cc2008)
    data['cc1975'] = data[['cc1975_lbond', 'cc1975_sbond', 'cc1975_loan', 'cc1975_gov']].dropna().sum(1)
    data['cc2002'] = data[['cc2002_bond', 'cc2002_loan', 'cc2002_gov']].dropna().sum(1)
    data['cc2008'] = data[['cc2008_bond', 'cc2008_loan', 'cc2008_gov']].dropna().sum(1)
    data['cc'] = data[['cc2008', 'cc2002', 'cc1975']].transform(Transform.link)    
    # GDP대비 기업신용 12개월 차분
    data['cc_gdp'] = (data['cc']/data['gdp'].rolling(12).sum()*3).rolling(3).mean()
    data['cc_gdp_diff'] = data['cc_gdp'].diff(12)
    
    ## 취약성/자산가격/부동산
    # KB 주택매매가격지수 3개월 이동평균 백분율 변화량
    data['kb_hp_pchg'] = data['kb_hp'].rolling(3).mean().pct_change()
    # KB 주택매매가격지수(수도권) 3개월 이동평균 백분율 변화량
    data['kb_hp_mt_pchg'] = data['kb_hp_mt'].rolling(3).mean().pct_change()
    # KB 주택매매가격지수(서울) 3개월 이동평균 백분율 변화량
    data['kb_hp_se_pchg'] = data['kb_hp_se'].rolling(3).mean().pct_change()
    # BIDAS/CS 전국
    data['bcs_ap_pchg'] = data['bcs_ap'].rolling(3).mean().pct_change()
    # BIDAS/CS 수도권
    data['bcs_ap_mt_pchg'] = data['bcs_ap_mt'].rolling(3).mean().pct_change()
    # BIDAS/CS 서울
    data['bcs_ap_se_pchg'] = data['bcs_ap_se'].rolling(3).mean().pct_change()
    # 미분양주택현황-서울
    data['uhi_se_pchg'] = data['uhi_se'].rolling(3).mean().pct_change()
    # 미분양주택현황-인천
    data['uhi_ic_pchg'] = data['uhi_ic'].rolling(3).mean().pct_change()
    # 미분양주택현황-경기
    data['uhi_gg_pchg'] = data['uhi_gg'].rolling(3).mean().pct_change()
    
    ## 트리거/금리
    # CD 스프레드 12개월 차분
    data['cd_sp_diff'] = data['cd_sp'].diff(12)
    # CP 스프레드 12개월 차분
    data['cp_sp_diff'] = data['cp_sp'].diff(12)
    # 국가신용 스프레드 12개월 차분
    data['sr_sp'] = data['kb10y'] - data['ub10y']
    data['sr_sp_diff'] = data['sr_sp'].diff(12)
    # (마이너스) 기간 프리미엄 스프레드 12개월 차분
    data['tp_sp_neg_diff'] = data['tp_sp_neg'].diff(12)
    # 회사채 AA 신용스프레드 12개월 차분
    data['cb_kb_diff'] = data['cr_sp'].diff(12)
    # 금융채 AA 스프레드 12개월 차분
    data['fi_kb'] = data['fiaa3y'] - data['kb3y']
    data['fi_kb_diff'] = data['fi_kb'].diff(12)
    
    ## 트리거/변동성
    # KOSPI 200 지수 CMAX
    data['stock_cmax'] = data['kospi2'].transform(Transform.cmax)
    # KOSDAQ 지수 CMAX
    data['kosdaq_cmax'] = data['kosdaq'].transform(Transform.cmax)
    # DOW JONES 지수 CMAX
    data['dow_cmax'] = data['dow'].transform(Transform.cmax)
    # 은행업 변동성
    data['bank_mv'] = data['stockbank'].transform(Transform.mvol)
    # 증권 변동성1
    data['secu_mv1'] = data['kossecu'].transform(Transform.mvol)
    # 증권 변동성2: 1이나 2 중 사용
    data['secu_mv2'] = data['stocksecu'].transform(Transform.mvol)
    data['secu_mv'] = data['secu_mv2']
    # 보험 변동성1
    data['ins_mv1'] = data['kosins'].transform(Transform.mvol)
    # 보험 변동성2: 1이나 2 중 사용
    data['ins_mv2'] = data['stockins'].transform(Transform.mvol)
    data['ins_mv'] = data['ins_mv2']
    # 미달러 환율 변동성
    data['er_mv'] = data['er'].transform(Transform.mvol)
    
    ## 트리거/경기
    # (마이너스) GDP 성장률
    data['gdp_growth_neg'] = -data['gdp_growth']
    # (마이너스) 경기종합지수(동행지수순환변동치) 3개월 이동평균 백분율 변화량
    data['cei_growth_neg'] = -data['cei'].rolling(3).mean().pct_change()
    # (마이너스) 전산업생산지수 3개월 이동평균 백분율 변화량
    data['ipi_growth_neg'] = -data['ipi'].rolling(3).mean().pct_change()
    
    ## 트리거/심리
    # EPU 3개월 이동평균 백분율 변화량
    data['epu_pchg'] = data['epu'].rolling(3).mean().pct_change()
    # EPU(trade policy) 3개월 이동평균 백분율 변화량
    data['epu_tp_pchg'] = data['epu_tp'].rolling(3).mean().pct_change()
    # (마이너스) 경제심리지수(순환변동치) 3개월 이동평균 백분율 변화량
    data['esi_pchg_neg'] = -data['esi'].rolling(3).mean().pct_change()
    # (마이너스) 뉴스심리지수 3개월 이동평균 백분율 변화량
    data['nsi_pchg_neg'] = -data['nsi'].rolling(3).mean().pct_change()

    return data

def get_crises(cfpi, k=1, horizon=6, group_bgn_ext=3, group_end_ext=3):
    """CFPI가 임계치를 넘는 기간을 기준으로, 위기기간 및 학습데이터의 그룹을 산정한다.

    Args:
        k: 위기 임계치(CFPI 표준편차의 배수)
        horizon: 위기기간 앞쪽에서 위기에 확장하여 포함할 기간(개월)
        group_bgn_ext: 확장된 위기기간 앞쪽에서 위기와 동일 그룹에 포함할 기간(개월)
        group_end_ext: 확장된 위기기간 뒷쪽에서 위기와 동일 그룹에 포함할 기간(개월)

    Returns:
        crises: 기본/확장/디레버리징(term/ext_term/post_term)위기 기간 및 학습그룹(group)정보
    """
    crises = cfpi.rename('cfpi').to_frame()
    # {k} 표준편차 이상의 CFPI에 대해 위기기간으로 지정
    crises['term'] = (cfpi > cfpi.std()*k).astype(int) # astype(int)해서 불리언을 1 또/는 0으로 할당/
    # 긱 위기기간의 직전 {horizon}개월을 확장 위기기간으로 지정 (예측 시계 확장)//
    crises['ext_term'] = crises['term'][::-1].rolling(horizon+1, min_periods=1).max()[::-1].astype(int)
    # 각 위기기간은 학습/예측시에 나눌 수 없는 하나의 그룹으로, 나머지는 각 데이터 포인트를 별개의 그룹으로 지정
    crises['groups'] = crises['ext_term'].rolling(2, min_periods=1).apply(lambda x: 0 if sum(x) == 2 else 1).cumsum()
    # 각 위기기간에서, CFPI가 정점을 찍은 후 내려오는 디레버리징 기간을 구분 (학습시 선택적으로 제외)
    is_post_term = lambda x: ((x > cfpi.std()*k) & (x.cummax() == x.max()) & (x.cummax() == x.cummax().shift(1))) * 1
    crises['post_term'] = crises.groupby('groups')['cfpi'].apply(is_post_term).set_axis(crises.index)
    # 각 확장 위기기간에 대해 이전 {group_bgn_ext}개월과 이후 {group_end_ext}개월을 동일 그룹으로 지정 (버퍼)
    if group_bgn_ext + group_end_ext > 0:
        get_group_scope = [lambda x: min(x.index)-group_bgn_ext, lambda x: max(x.index)+group_end_ext]
        group_scope = crises[crises['ext_term'] == 1].groupby('groups')['ext_term'].agg(get_group_scope)
        prev_scope = crises.index.min()
        for group, scope in group_scope.iterrows():
            assert prev_scope < scope[0]
            crises.loc[scope[0]:scope[1], 'groups'] = group
            prev_scope = scope[1]
    return crises

def get_crises2(cfpi, k_enter=1.0, k_exit=0.8, horizon=6, enter=2, exit=1, group_bgn_ext=3, group_end_ext=3):
    """
    CFPI가 임계치를 넘는 기간을 기준으로, 위기기간 및 학습데이터의 그룹을 산정한다.

    Args:
        k: 위기 임계치(CFPI 표준편차의 배수)
        horizon: 위기기간 앞쪽에서 위기에 확장하여 포함할 기간(개월)
        enter: 연속 초과(window) 기간(개월), 모두 초과 시 term = 1
        exit: 위기 종료를 위해 연속 미초과되어야 하는 기간(개월)
        group_bgn_ext: 확장된 위기기간 앞쪽에서 위기와 동일 그룹에 포함할 기간(개월)
        group_end_ext: 확장된 위기기간 뒷쪽에서 위기와 동일 그룹에 포함할 기간(개월)

    Returns:
        pd.DataFrame crises: 기본/확장/디레버리징(term/ext_term/post_term) 위기 기간 및 학습그룹(group) 정보
    """
    
    crises = cfpi.rename('cfpi').to_frame()

    # 진입, 탈출 임계치 계산
    sigma = cfpi.std()
    mean = cfpi.mean()
    thresh_enter = mean + sigma * k_enter
    thresh_exit  = mean + sigma * k_exit

    # 히스테리시스 버퍼 기반 위기기간(term) 식별
    term = pd.Series(False, index=cfpi.index)
    in_crisis = False
    consec_exceed = 0
    consec_below = 0
    idx = cfpi.index
    
    for t, val in cfpi.items():
        pos = idx.get_indexer([t])[0]
        if not in_crisis:
            if val > thresh_enter:
                consec_exceed += 1
            else:
                consec_exceed = 0
            # enter 개월 연속 초과 시 진입
            if consec_exceed >= enter:
                in_crisis = True
                # 처음 초과 시점부터 마크
                start = max(0, pos - enter + 1)
                term.iloc[start:pos+1] = True
                consec_below = 0
        else:
            if val < thresh_exit:
                consec_below += 1
            else:
                consec_below = 0
            # exit 개월 연속 미초과 시 탈출
            if consec_below >= exit:
                in_crisis = False
                consec_exceed = 0
        # 위기 상태면 현재 시점에도 마크
        if in_crisis:
            term.iloc[pos] = True

    crises['term'] = term.astype(int)

    # 확장 위기기간 식별(horizon 개월 포함)
    ext = crises['term'][::-1].rolling(horizon+1, min_periods=1).max()[::-1]
    crises['ext_term'] = ext.astype(int)

    # groups: ext_term=1 구간을 같은 그룹으로
    groups = crises['ext_term'].rolling(2, min_periods=1) \
                   .apply(lambda x: 0 if x.sum()==2 else 1) \
                   .cumsum().astype(int)
    crises['groups'] = groups

    # post_term: 그룹별 peak 이후 위기 구간
    def find_post(sub):
        cfpi_vals = sub['cfpi'].values
        term_vals = sub['term'].values.astype(bool)
        # peak position in sub
        peak_pos = cfpi_vals.argmax()
        post = [False] * len(cfpi_vals)
        for i in range(len(cfpi_vals)):
            if i > peak_pos and term_vals[i]:
                post[i] = True
        return pd.Series(post, index=sub.index).astype(int)
    crises['post_term'] = crises.groupby('groups').apply(find_post).droplevel(0)

    # ext_term 구간 앞뒤 확장
    if group_bgn_ext + group_end_ext > 0:
        for grp, sub in crises[crises['ext_term']==1].groupby('groups'):
            start = sub.index.min() - group_bgn_ext
            end   = sub.index.max() + group_end_ext
            crises.loc[start:end, 'groups'] = grp

    return crises

def plot_cfpi(cfpi, gdp_growth, k, horizon=6, 
              group_bgn_ext=3, group_end_ext=3, xlim=['1999-01', '2024-01'], ylim=None, figsize=(12, 6)):
    """CFPI를 위기기간, GDP성장률과 함께 차트에 그린다.

    Args:
        cfpi: CFPI 데이터
        gdp_growth: GDP성장률 데이터
        k: 위기기간 산정을 위한 CFPI 임계치
        horizon: 위기기간 예측 시계(월)
        xlim: 표시 기간
    """
    crises = get_crises(cfpi, k, horizon, group_bgn_ext, group_end_ext)
    _, ax = plt.subplots(figsize=figsize)
    # CFPI 및 위기기간 차트 표시
    render_crises(crises, ax, True)
    # GDP성장률 및 국소최저점 차트 표시
    render_gdp_drop(gdp_growth, freq='M', ax=ax)
    # x축 설정
    ax.tick_params(axis='x', labelsize=15, which='both')
    ax.set_xlabel('')
    # y축(이중축) 설정
    ax.tick_params(axis='y', labelsize=15)
    ax_right = ax.twinx()
    ax_right.tick_params(axis='y', labelsize=15)
    ax_right.set_ylim(ax.get_ylim())
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
        
        
def plot_cfpi2(cfpi, gdp_growth, k_enter=1.0, k_exit=0.8, horizon=6, enter=2, exit=1, 
               group_bgn_ext=3, group_end_ext=3, xlim=['1999-01', '2024-01'], ylim=None, figsize=(12, 6)):
    """CFPI를 위기기간, GDP성장률과 함께 차트에 그린다.

    Args:
        cfpi: CFPI 데이터
        gdp_growth: GDP성장률 데이터
        k: 위기기간 산정을 위한 CFPI 임계치
        horizon: 위기기간 예측 시계(월)
        xlim: 표시 기간
    """
    crises = get_crises2(cfpi, k_enter, k_exit, horizon, enter, exit, group_bgn_ext, group_end_ext)
    _, ax = plt.subplots(figsize=figsize)
    # CFPI 및 위기기간 차트 표시
    render_crises2(crises, k_enter, k_exit, ax, ylim, True)
    # GDP성장률 및 국소최저점 차트 표시
    render_gdp_drop(gdp_growth, freq='M', ax=ax)
    # x축 설정
    ax.tick_params(axis='x', labelsize=15, which='both')
    ax.set_xlabel('')
    # y축(이중축) 설정
    ax.tick_params(axis='y', labelsize=15)
    ax_right = ax.twinx()
    ax_right.tick_params(axis='y', labelsize=15)
    ax_right.set_ylim(ax.get_ylim())
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if ylim is None:
        ymin, ymax = ax.get_ylim()
        ax.set_ylim([ymin, ymax])


def render_crises(crises, ax=None, cfpi_plot=False, crises_plot=True):
    """CFPI 및 위기기간을 차트에 표시한다.

    Args:
        crises: 위기기간 데이터
        ax: 차트축(Axes) 개체
        cfpi_plot: CFPI 표시 여부
        crises_plot: 위기기간 표시 여부
    """
    cfpi = crises.cfpi
    if cfpi_plot:
        # CFPI 라인차트
        if ax is None:
            ax = cfpi.plot(color='black')
        else:
            cfpi.plot(ax=ax, color='black')
        # y={0, CFPI임계치} 기준선 표시
        ax.axhline(y=0, linestyle='-', color='black')
        ax.axhline(y=crises[crises.term == 1].cfpi.min(), linestyle=':', color='black')
        
    if ax is None:
        _, ax = plt.subplots()
    if crises_plot:
        ymin, ymax = ax.get_ylim()
        ax.set_ylim([ymin, ymax])
        # 예측시계(red), 위기기간중 상승(red+red)/하강(red+red+yellow) 구간을 색의 중첩으로 구분
        ax.fill_between(cfpi.index, ymin, ymax,color='red', where=crises.ext_term, alpha=0.1)
        ax.fill_between(cfpi.index, ymin, ymax, color='red', where=crises.term, alpha=0.2)
        ax.fill_between(cfpi.index, ymin, ymax, color='yellow', where=crises.post_term, alpha=0.1)
        
        
def render_crises2(crises, k_enter=1.0, k_exit=0.8, ax=None, ylim=None, 
                   cfpi_plot=False, crises_plot=True, ispred=False, shared_index=None):
    """CFPI 및 위기기간을 차트에 표시한다.

    Args:
        crises: 위기기간 데이터
        k_enter: 위기기간 진입 임계치
        k_exit: 위기기간 탈출 임계치
        ax: 차트축(Axes) 개체
        ylim: y축 범위
        cfpi_plot: CFPI 표시 여부
        crises_plot: 위기기간 표시 여부
    """
    import matplotlib.patches as mpatches
    
    cfpi = crises.cfpi
    thrs_enter = cfpi.std() * k_enter
    thrs_exit = cfpi.std() * k_exit
    
    if cfpi_plot:
        # CFPI 라인차트
        if ax is None:
            ax = cfpi.plot(color='black')
        else:
            cfpi.plot(ax=ax, color='black')
        # y={0, CFPI임계치} 기준선 표시
        ax.axhline(y=0, linestyle='-', color='black')
        ax.axhline(y=thrs_enter, linestyle='--', color='black')
        ax.axhline(y=thrs_exit, linestyle=':', color='grey')
    if ax is None:
        _, ax = plt.subplots()
        
    if ispred is False:  
        if crises_plot and ylim is not None:
            ymin, ymax = ylim[0], ylim[1]
            ax.set_ylim([ymin, ymax])
            # 예측시계(red), 위기기간중 상승(red+red)/하강(red+red+yellow) 구간을 색의 중첩으로 구분
            ax.fill_between(cfpi.index, ymin, ymax, color='darkorange', where=crises.ext_term, alpha=0.3, label='Warning') # (crises.ext_term == 1) & (crises.term != 1)
            ax.fill_between(cfpi.index, ymin, ymax, color='orangered', where=crises.term, alpha=0.5, label='Crisis') # (crises.term == 1) & (crises.post_term != 1)
            ax.fill_between(cfpi.index, ymin, ymax, color='skyblue', where=crises.post_term, alpha=0.3, hatch='///', label='Relief') # (crises.post_term == 1), sienna
            
            legend_elements = [
                mpatches.Patch(color='darkorange', alpha=0.3, label='주의'),
                mpatches.Patch(color='orangered', alpha=0.4, label='위기(경보단계)'),
                mpatches.Patch(color='sienna', alpha=0.3, hatch='///', label='위기(완화단계)')
            ]
            #ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=True, fontsize=10)
            
        if crises_plot and ylim is None:
            ymin, ymax = ax.get_ylim()
            ax.set_ylim([ymin, ymax])
            # 예측시계(red), 위기기간중 상승(red+red)/하강(red+red+yellow) 구간을 색의 중첩으로 구분
            ax.fill_between(cfpi.index, ymin, ymax,color='red', where=crises.ext_term, alpha=0.1)
            ax.fill_between(cfpi.index, ymin, ymax, color='red', where=crises.term, alpha=0.2)
            ax.fill_between(cfpi.index, ymin, ymax, color='yellow', where=crises.post_term, alpha=0.1)
    else:
        if crises_plot and ylim is not None and shared_index is not None:
            ymin, ymax = ylim[0], ylim[1]
            ax.set_ylim([ymin, ymax])
            
            # shared_index 기반으로 정수 x축 생성
            x_coords = range(len(shared_index))
            
            # shared_index에 해당하는 위기기간 데이터 추출
            ext_mask = []
            term_mask = []
            post_mask = []
            
            for period in shared_index:
                if period in crises.index:
                    ext_mask.append(bool(crises.loc[period, 'ext_term']))
                    term_mask.append(bool(crises.loc[period, 'term']))
                    post_mask.append(bool(crises.loc[period, 'post_term']))
                else:
                    ext_mask.append(False)
                    term_mask.append(False)
                    post_mask.append(False)
            
            # 정수 x축으로 위기기간 표시
            ax.fill_between(x_coords, ymin, ymax, color='darkorange', where=ext_mask, alpha=0.3, label='Warning')
            ax.fill_between(x_coords, ymin, ymax, color='orangered', where=term_mask, alpha=0.5, label='Crisis')
            ax.fill_between(x_coords, ymin, ymax, color='skyblue', where=post_mask, alpha=0.3, hatch='///', label='Relief')
        



def render_gdp_drop(gdp_growth, freq='Q', ax=None, hgrid=False):
    """GDP 성장률 및 국소최저점(local minima)을 차트에 표시한다.

    Args:
        gdp_growth: GDP성장률 데이터
        freq: 표시할 빈도의 단위(M, Q)
        ax: 차트축(Axes) 개체
        hgrid: y=0축 표시 여부
    """
    # 저점이 과거 1년 평균보다 1표준편차 이상 떨어진 경우 local minima로 설정
    loc_min = lambda x, t=0, q=4: x[(x.shift(1) > x) & (x.shift(-1) > x) &
                                    (t > x) & (x.rolling(q).mean() - x.std() > x)]
    if ax is None:
        ax = gdp_growth.resample(freq).ffill().plot()
    if hgrid:
        ax.axhline(y=0, linestyle=':')
    # GDP성장률이 국소최저점을 지나는 시점마다 차트에 세로축 표시
    gdp_drops = gdp_growth.resample('Q').mean().agg(loc_min).index
    for date in gdp_drops.asfreq(freq):
        ax.axvline(x=date, color='k', linestyle=':', linewidth=3)
        
#def plot_cfpi_fsi


# II. 조기경보모형 하이퍼파라미터 튜닝 및 평가

class SignalExtraction(BaseEstimator, ClassifierMixin):
    """(Scikit-learn 호환) 신호추출법 모형의 구현체"""
    
    def __init__(self, significance=0.75):
        # 허용 가능한 NSR의 최대값 설정
        self.significance = significance

    def fit(self, X, y, **kwargs):
        """주어진 데이터셋을 학습한다."""
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        self.cutoffs = []
        self.weights = []
        # 입력변수별로 적용할 임계치와 가중치를 산정
        for var in range(X.shape[1]):
            # NSR을 최소화하는 변수값의 임계치(cutoff) 도출
            var_values = np.unique(X[:, var])
            nsrs = [get_perf(y, X[:, var], var_value)['nsr'] for var_value in var_values]
            cutoff = np.nanargmin(nsrs)
            self.cutoffs.append(var_values[cutoff])
            # 최소 NSR이 {significance} 이하이면 NSR의 역수를 가중치로 사용, 이상이면 무시
            self.weights.append(1/nsrs[cutoff] if (nsrs[cutoff] < self.significance) else 0)
        norm = sum(self.weights)
        self.weights = [weight / norm for weight in self.weights]
        return self

    def predict(self, X):
        """고정값(0.5)을 기준으로 예측값을 0, 1로 나눈다."""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5) * 1

    def predict_proba(self, X):
        """변수별로 각 임계치를 넘는 값에 대해 각 가중치를 곱하고 합하여 예측값을 산정한다."""
        check_is_fitted(self)
        X = check_array(X)
        proba = ((X > self.cutoffs) * self.weights).sum(axis=1)
        return np.stack([1 - proba, proba], 1)


def get_perf(Y, Y_pred, threshold=0.15, calc_auc=False): # 기존 threshold=0.5
    """이진 분류 모델의 예측값과 실제값을 비교하여 예측성능을 평가한다.

    Args:
        Y: 분류 실제값
        Y_pred: 분류 예측값
        threshold: 분류 임계치
        calc_auc: AUC 점수 계산 여부

    Returns:
        perf: 성능평가기준(acc, nsr, f1, auc 등)별 수치
    """
    # 입력받은 분류 임계치를 기준으로 confusion matrix 구성
    Actl = np.array(Y, dtype='bool')
    Pred = Y_pred >= threshold
    tp = np.logical_and( Actl,  Pred).sum()
    tn = np.logical_and(~Actl, ~Pred).sum()
    fp = np.logical_and(~Actl,  Pred).sum()
    fn = np.logical_and( Actl, ~Pred).sum()
    perf = {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}
    # 정확도(accuracy)
    perf['acc'] = float(tp + tn) / float(tp + tn + fp + fn) if ((tp + tn + fp + fn > 0))  else np.nan
    # 재현율/민감도(recall, sensitivity) = true positive rate = (1 - false negative rate)
    perf['tpr'] = float(tp) / float(tp + fn)        if ((tp + fn > 0))  else np.nan
    # 1종오류율 = false positive rate = (1 - true negative rate)
    perf['fpr'] = 1 - float(tn) / float(tn + fp)    if (tn + fp > 0)    else np.nan
    # NSR(Noise-to-Signal) 비율 = 1종오류율 / 재현율 = 1종오류율 / (1 - 2종오류율)
    perf['nsr'] = perf['fpr'] / perf['tpr']         if ((perf['fpr'] > 0) & (perf['tpr'] > 0)) else np.nan
    # F1 점수 = 정밀도(TP/(TP+FP))와 재현율의 조화평균
    perf['f1']  = 2 * tp / (2 * tp + fp + fn)       if (tp + fp + fn > 0)    else np.nan
    # ROC-AUC(Receiver Operating Characteristic - Area Under the Curve) 점수
    if calc_auc:
        perf['auc'] = roc_auc_score(Actl, Y_pred)   if ((tp + fn > 0) & (tn + fp > 0)) else np.nan
    return perf


def run_cv(models, model_param_grids, X, y, exclude_post_term=True,
           cv_method='sgkf', # cv_method='sgkf' or 'loeo'
           n_splits=5,
           cv_inner_method='sgkf', # cv_inner_method='sgkf' or 'loeo'
           inner_splits=4,
           crises_groups=None
           ):
    """각 모델에 대해 하이퍼파라미터 튜닝한 후 예측결과를 산출한다.

    Args:
        models: (Scikit-learn 호환) 모델 목록
        model_param_grids: 하이퍼파라미터의 grid 탐색을 위한 사전 설정값 목록
        X: 입력변수
        y: 라벨 데이터
        exclude_post_term: 위기기간중 하강구간의 학습데이터 제외 여부
    """
    train_exclusion = y.post_term if exclude_post_term else None
    
    # 교차검증용 학습/테스트 데이터셋 생성
    folds = load_folds(X, y.ext_term, y.groups, train_exclusion=train_exclusion, 
                       cv_method=cv_method, n_splits=n_splits, crises_groups=crises_groups)
    print(f"Generated {len(folds)} folds(episodes) for cross-validation.")
    
    # 그리드 탐색을 통해 모델별 최적 하이퍼파라미터 산출
    best_params = grid_search_folds(folds, models, model_param_grids, 
                                    scoring='roc_auc', cv_inner_method=cv_inner_method, inner_splits=inner_splits)
    print('best_params = ', best_params)
    
    # 최적 하이퍼파라미터를 바탕으로 모델별 학습 및 예측
    results = train_and_test_folds(folds, models, best_params)
    
    # 모델별 예측성능을 표시
    summarize_results(results, models)
    
    return best_params, results


def load_folds(X, y, groups, train_exclusion=None, 
               cv_method='sgkf', n_splits=5, bootstrap=True, crises_groups=None): #n_splits=5
    """stratified group K-fold 교차검증을 위한 학습/테스트 데이터셋을 생성한다.

    Args:
        X: 입력변수
        y: 라벨 데이터
        groups: 교차검증 그룹 id 목록
        train_exclusion: 데이터포인트별 학습 데이터 제외여부(0:포함, 1:제외) 목록
        n_splits: 교차검증 폴드 개수
        bootstrap: 부트스트래핑 여부

    Returns:
        folds: 생성된 교차검증 데이터셋
    """
    folds = []
    
    if cv_method == 'loeo':
        bootstrap = False
        
    if cv_method == 'sgkf':
        splitter = StratifiedGroupKFold(n_splits=n_splits)
    elif cv_method == 'loeo':
        # Leave-One-Group-Out 교차검증
        splitter = LeaveOneGroupOut()
    else:
        raise ValueError("cv_method must be 'sgkf' or 'loeo'")
    

    for (train_idx, test_idx) in splitter.split(X, y, groups):
        if cv_method == 'loeo' and crises_groups is not None:
            test_grp =groups[test_idx][0]
            if test_grp not in crises_groups:
                continue
        
        
        # 위기기간중 디레버리징 구간 등 학습에서 제외할 구간이 있으면 제외
        if train_exclusion is not None:
            train_idx = np.array([idx for idx in train_idx if train_exclusion[idx] == 0])
            
        # 학습데이터의 class imbalance 완화를 위해 minor class 데이터를 부트스트래핑/업샘플링
        if bootstrap: # bootstrap / upsampling (sgkf only)
            pos_idx = train_idx[y[train_idx] == 1]
            neg_idx = train_idx[y[train_idx] == 0]
            major_cls = neg_idx if len(neg_idx) > len(pos_idx) else pos_idx
            minor_cls = pos_idx if len(neg_idx) > len(pos_idx) else neg_idx
            train_idx = np.concatenate((np.random.choice(major_cls, size=len(major_cls), replace=False),
                                        np.random.choice(minor_cls, size=len(major_cls), replace=True)))
        fold = {}
        # 학습/테스트 데이터셋별로 표준화
        scaler = StandardScaler()
        fold['train_X'] = scaler.fit_transform(X.iloc[train_idx])
        fold['train_y'] = y[train_idx]
        fold['train_groups'] = groups[train_idx] # 중첩 교차검증시 필요
        fold['test_X'] = scaler.transform(X.iloc[test_idx])
        fold['test_y'] = y[test_idx]
        fold['test_idx'] = test_idx # 데이터포인트별 테스트 결과 저장시 필요
        fold['test_index'] = X.index[test_idx]
        folds.append(fold)
        
    return folds


def grid_search_folds(folds, models, model_param_grids, 
                      scoring='roc_auc', cv_inner_method='sgkf', inner_splits=4):
    """stratified group K-fold 교차검증으로 모델별 성능평가와 최적 하이퍼파라미터 탐색을 수행한다.

    Args:
        folds: 교차검증 데이터셋
        models: (Scikit-learn 호환) 모델 목록
        model_param_grids: 하이퍼파라미터의 grid 탐색을 위한 사전 설정값 목록
        scoring: 하이퍼파라미터 튜닝시 성능평가 기준(e.g. roc_auc, f1, accuracy)

    Returns:
        best_params: 각 모델별 최적 하이퍼파라미터값
    """
    best_params = {}
    # 각 모델의 성능을 평가
    for model in tqdm(models):
        best_param = {}
        best_score = 0
        classifier = models[model]
        
        for f, fold in enumerate(folds):
            
            # 각 폴드내 중첩(nested) 교차검증을 통한 하이퍼파라미터 튜닝
            if cv_inner_method == 'loeo':
                inner_cv = LeaveOneGroupOut().split(fold['train_X'], fold['train_y'], fold['train_groups'])
            else:
                inner_cv = StratifiedGroupKFold(n_splits=inner_splits).split(fold['train_X'], fold['train_y'], fold['train_groups'])
                
            # 각 폴드내 중첩(nested) 교차검증을 통한 하이퍼파라미터 튜닝
            # splitter = StratifiedGroupKFold(n_splits=4) #n_splits=4
            # cv_inner = splitter.split(fold['train_X'], fold['train_y'], fold['train_groups'])
            grid = GridSearchCV(classifier, cv=inner_cv, param_grid=model_param_grids[model],
                                scoring=scoring, verbose=False, error_score='raise')
            grid.fit(fold['train_X'], fold['train_y'])
            print('%s %d - Best Score: %f, Best Params: %s' % (model, f, grid.best_score_, grid.best_params_))
            
            if grid.best_score_ > best_score:
                best_param = grid.best_params_
                best_score = grid.best_score_
        best_params[model] = best_param
        
    return best_params


def train_and_test_folds(folds, models, model_params=None):
    """하이퍼파라미터를 고정한 모델별로 교차검증 데이터셋에 대해 학습 및 테스트를 수행한다."""
    
    all_test_idx = np.concatenate([fold['test_index'] for fold in folds])
    
    # results = pd.DataFrame(index=np.arange(np.sum([len(fold['test_y']) for fold in folds])),
    #                        columns=(['fold', 'actl'] + [model for model in models]))
    
    results = pd.DataFrame(index=all_test_idx, columns= ['fold', 'actl'] + list(models.keys()))
    
    for f, fold in enumerate(folds):
        idx = fold['test_index']
        results.loc[idx, 'fold'] = f
        results.loc[idx, 'actl'] = fold['test_y']
        # results['fold'].iloc[fold['test_idx']] = f
        # results['actl'].iloc[fold['test_idx']] = fold['test_y']
        
    for model in tqdm(models):
        classifier = models[model]
        if model_params is not None:
            model_param = model_params[model]
            classifier.set_params(**model_param)
        for fold in folds:
            # 학습
            sample_weight = compute_sample_weight('balanced', fold['train_y'])
            
            if isinstance(classifier, Pipeline):
                classifier.fit(fold['train_X'], fold['train_y'], **{'clf__sample_weight': sample_weight})
            else:
                classifier.fit(fold['train_X'], fold['train_y'], sample_weight=sample_weight)
            # 테스트
            idx = fold['test_index']
            prob = classifier.predict_proba(fold['test_X'])
            results.loc[idx, model] = prob[:, 1] if len(prob.shape) > 1 else [prob[1]]
            # prob = classifier.predict_proba(fold['test_X'])
            # results[model].iloc[fold['test_idx']] = prob[:, 1] if len(prob.shape) > 1 else [prob[1]]
            
    return results


def summarize_results(results, models):
    """분류예측결과를 입력받아 모델별 예측성능을 계산하여 표시한다."""
    perfs = []
    for model in models:
        missing = results[model].isna()
        perf = get_perf(results[~missing]['actl'], results[~missing][model], calc_auc=True)
        perfs.append(perf)
    print(pd.DataFrame(perfs, index=models))


def plot_roc_curve(preds, agg=False):
    """각 모델별 분류예측 결과를 바탕으로 ROC 곡선을 그린다.

    Args:
        results: 각 컬럼은 fold(폴드 번호), actl(라벨값)을 제외하고 모델명(예측값)으로 구성
    """
    model_names = [model_name for model_name in preds.columns if model_name not in ['fold', 'actl']]
    # 범례에 표시할 모델별 ROC-AUC점수를 fold별로 집계/평균하여 산정
    agg_auc = lambda x: pd.DataFrame([roc_auc_score(list(x.actl), x[model]) for model in model_names],
                                     index=model_names).T
    agg_perf = pd.DataFrame(preds.groupby('fold').apply(agg_auc).mean()).T
    # ROC 곡선 표시
    actl = preds['actl'].to_list()
    plt.figure(figsize=(6, 6))
    for model in model_names:
        fpr, tpr, _ = roc_curve(actl, preds[model])
        auc_score = agg_perf[model] if agg else roc_auc_score(list(actl), preds[model]) # agg옵션에 따라 폴드별 auc 평균 또는 전체 auc 산출
        plt.plot(fpr, tpr, lw=3, alpha=0.5, label='%s (avg. auc=%0.2f)' % (model, auc_score))
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, 1.01])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('Receiver Operating Characteristic', fontsize=16)
    plt.legend(loc='lower right', fontsize='small', prop={'size': 14}, handlelength=2, handletextpad=0.5, labelspacing=0.5)
    plt.show()


# III. 조기경보모형의 활용

class EarlyWarningModel:
    """조기경보모형의 학습, 실행, 저장, 로딩을 위한 클래스"""

    def __init__(self, model=None):
        self.model = model
        self.scaler = StandardScaler()

    def train(self, X, y):
        """학습데이터를 기준으로 scaler와 model의 파라미터를 업데이트한다."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def scale(self, X):
        """학습한 데이터와 동일한 기준으로 신규 데이터를 표준화한다."""
        return self.scaler.transform(X)

    def predict(self, X, decompose=False):
        """학습된 모델로 신규 데이터를 예측하고, 변수별 기여도를 분해한다.

        Args:
            X: 입력 변수
            decompose: 기여도 분해 여부

        Returns:
            results: 예측결과
            impacts: 변수별 기여도 분해 결과
        """
        results = pd.DataFrame(columns=['period', 'pred']).set_index('period')
        impacts = pd.DataFrame(columns=['period', 'variable', 'impact']).set_index(['period', 'variable'])
        X_scaled = self.scale(X)
        for time_idx, time in enumerate(X.index):
            X_original = X_scaled[time_idx, :].reshape(1, -1)
            original_value = self.model.predict_proba(X_original)[:, 1][0]
            period = pd.to_datetime(time.strftime('%Y-%m'))
            results.loc[period] = original_value
            # 변수별 기여도 분해
            if decompose and time_idx > 0:
                for var_idx, var in enumerate(X.columns):
                    X_modified = np.copy(X_original)
                    X_modified[0, var_idx] = X_scaled[time_idx-1, var_idx]
                    modified_value = self.model.predict_proba(X_modified)[:, 1][0]
                    impact = original_value - modified_value
                    impacts.loc[(period, var), :] = impact
        return results, impacts
    
    def predict_weekly(self, X, decompose=False):
        
        weekly_results = pd.DataFrame(columns=['pred'])
        weekly_results.index.name = 'period'

        weekly_impacts = pd.DataFrame(columns=[0])
        weekly_impacts.index = pd.MultiIndex.from_tuples([], names=['period', 'variable'])
        
        # 모델의 scale 메서드로 데이터 표준화
        X_scaled = self.scale(X)

        # 각 시점별로 예측 및 기여도 계산
        for time_idx, vintage_time in enumerate(X.index):
            X_original = X_scaled[time_idx, :].reshape(1, -1)
            original_value = self.model.predict_proba(X_original)[:, 1][0]
            
            # 예측 결과 저장 (빈티지 시점 그대로 사용)
            weekly_results.loc[vintage_time, 'pred'] = original_value
            
            # 변수별 기여도 분해 (이전 시점과 비교)
            if time_idx > 0:
                for var_idx, var in enumerate(X.columns):
                    X_modified = np.copy(X_original)
                    X_modified[0, var_idx] = X_scaled[time_idx-1, var_idx]  # 이전 시점 값으로 변경
                    modified_value = self.model.predict_proba(X_modified)[:, 1][0]
                    impact = original_value - modified_value
                    weekly_impacts.loc[(vintage_time, var), 0] = impact
        return weekly_results, weekly_impacts

    def save(self, id=None):
        """학습된 model과 scaler를 저장한다."""
        with open(f'{id}_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        with open(f'{id}_scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)

    def load(self, id=None):
        """학습된 model과 scaler를 불러온다."""
        with open(f'{id}_model.pkl', 'rb') as f:
            self.model = pickle.load(f)
        with open(f'{id}_scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)


def plot_predicted(results, crises=None, perc70=None, perc90=None,
                   line_styles=['-', '--', '-.', ':']):
    """조기경보모형의 예측결과를 차트에 표시한다.

    Args:
        results: 예측결과
        crises: 위기기간 데이터(지정한 경우만 표시)
        perc70: 기준 예측치의 70분위 수치(지정한 경우만 표시)
        perc90: 기준 예측치의 90분위 수치(지정한 경우만 표시)
        line_styles: 모델이 여러개인 경우 차트를 구분하기 위한 라인 스타일
    """
    _, ax = plt.subplots(figsize=(9, 4))
    # 모델별 라인차트
    for i, (model_name, result) in enumerate(results.items()):
        if model_name == 'ET':
            marker = 'o'
        else:
            marker = None
        ax.plot(result.index, result, label=model_name, color='black',
                linestyle=line_styles[i], marker=marker, linewidth=2) # line_styles[i]
    # x축 설정
    ax.set_xlim(result.index.min(), result.index.max())
    ax.set_xticklabels([])
    # y축 설정
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.tick_params(axis='y', labelsize=15)
    # 범례 설정
    ax.legend(loc='upper left', frameon=False, fontsize=18)
    # 위기기간 표시
    if crises is not None:
        render_crises(crises, ax=ax)
    # 70분위, 90분위 기준선 표시
    if perc70 is not None:
        ax.axhline(perc70, color='dimgrey', linestyle='--', lw=1.5)
    if perc90 is not None:
        ax.axhline(perc90, color='dimgrey', linestyle=':', lw=1.5)


def plot_decomposed(impacts, feature_ids, feature_groups,
                    palette=DEFAULT_PALETTE, bar_width=0.4, legend_row=2, legend_col=3):
    """조기경보모형의 예측결과를 바탕으로 변수별 기여도를 차트에 표시한다.

    Args:
        impacts: 변수별 기여도 데이터
        feature_ids: 입력 변수명 목록
        feature_groups: 입력 변수 그룹명 목록
        palette: 변수 그룹별 색상
        bar_width: 단일 바 넓이
        legend_row: 범례 행 개수
        legend_col: 범례 열 개수
    """
    _, ax = plt.subplots(figsize=(9, 4))
    periods = impacts.index.get_level_values(0).unique()
    # 바차트 - 각 일자/변수 그룹별로 양/음의 영향도를 더하여 표시
    for period_idx, period in enumerate(periods):
        bottom_pos = 0
        bottom_neg = 0
        for feature_id, feature_group in zip(feature_ids, feature_groups):
            impact = impacts.loc[(period, feature_id)][0]
            if impact > 0:
                ax.bar(period_idx+0.5, impact, bottom=bottom_pos, color=palette[feature_group], width=bar_width)
                bottom_pos += impact
            else:
                ax.bar(period_idx+0.5, impact, bottom=bottom_neg, color=palette[feature_group], width=bar_width)
                bottom_neg += impact
    # x축 설정
    ax.set_xlim([0 - bar_width/2, len(periods) - bar_width/2])
    ax.set_xticks([i+0.5 for i in range(len(periods))])
    xticklabels = [period.strftime('%b\n%y') if period.month in [3, 6, 9, 12] else '' for period in periods]
    ax.set_xticklabels(xticklabels, fontsize=18)
    # y축 설정
    y_top = impacts[impacts >= 0].groupby('period').sum().max()[0]
    y_bottom = impacts[impacts < 0].groupby('period').sum().min()[0]
    ax.set_ylim([max(y_bottom*1.2, -1), min(y_top*1.2, 1)])
    ax.tick_params(axis='y', labelsize=18)
    # y=0 기준선 표시
    ax.axhline(0, color='black', linewidth=1)
    # 범례 설정
    labels = np.array([label for label in palette]).reshape(legend_row, legend_col).T.flatten()
    handles = [plt.Rectangle((0, 0), 1, 1, color=palette[label]) for label in labels]
    ax.legend(handles, labels, loc='lower center', frameon=False, ncol=legend_col,
              bbox_to_anchor=(0.5, -0.55), fontsize=18, columnspacing=0.5)


def plot_pred_decomp(results, impacts, feature_ids, feature_groups, crises=None, 
                     perc70=None, perc90=None, line_styles=['-', '--', '-.', ':'],
                     palette=DEFAULT_PALETTE, bar_width=0.4, legend_row=1, legend_col=8,
                     ax1_ylim=[0, 1], figsize=(9, 7), fig_name='pred_decomp', 
                     print_text=True, is_save_fig=True):
    
    """조기경보모형의 예측결과와 변수별 기여도를 하나의 figure에 표시한다.

    Args:
        results: 예측결과
        impacts: 변수별 기여도 데이터
        feature_ids: 입력 변수명 목록
        feature_groups: 입력 변수 그룹명 목록
        crises: 위기기간 데이터(지정한 경우만 표시)
        perc70: 기준 예측치의 70분위 수치(지정한 경우만 표시)
        perc90: 기준 예측치의 90분위 수치(지정한 경우만 표시)
        line_styles: 모델이 여러개인 경우 차트를 구분하기 위한 라인 스타일
        palette: 변수 그룹별 색상
        bar_width: 단일 바 넓이
        legend_row: 범례 행 개수
        legend_col: 범례 열 개수
        figsize: figure 크기
        fig_name: 저장할 figure 파일명
        print_text: 하단에 요인분해 결과 텍스트 추가 여부
    """
    
    color = ['darkblue', 'grey', 'black', 'red', 'green', 'yellow', 'purple', 'orange', 'brown']
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 2]})
    
    periods = impacts.index.get_level_values(0).unique()
    
    # periods가 비어있는 경우 예외처리
    if len(periods) == 0:
        print("Warning: No periods found in impacts data. Skipping decomposition plot.")
        ax1.text(0.5, 0.5, 'No impact data available', ha='center', va='center', transform=ax1.transAxes)
        ax2.text(0.5, 0.5, 'No impact data available', ha='center', va='center', transform=ax2.transAxes)
        plt.tight_layout()
        if is_save_fig:
            plt.savefig(f'figure/{fig_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
        return
    
    # 인덱스 타입에 따라 적절히 처리
    if not isinstance(periods[0], pd.Timestamp):
        try:
            # 문자열이면 datetime으로 변환 시도
            shared_index = pd.to_datetime(sorted(set(periods).intersection(results[next(iter(results))].index)))
        except:
            # 변환 실패시 원본 사용
            shared_index = sorted(set(periods).intersection(results[next(iter(results))].index))
    else:
        # 이미 datetime이면 그대로 사용
        shared_index = sorted(set(periods).intersection(results[next(iter(results))].index))
    
    # 상단: 예측결과 차트
    for i, (model_name, result) in enumerate(results.items()):
        
        marker = 'o' if model_name == 'ET' or model_name == 'ET(은행)' or model_name == 'ET(비은행)' or model_name == 'ET_baseline' else None
        #marker = None
        x_values = result.loc[shared_index].index
        y_values = result.loc[shared_index]["pred"]
        #ax1.plot(result.index, result, label=model_name, color='black',
        #        linestyle=line_styles[i], marker=marker, linewidth=2)
        ax1.plot(range(len(x_values)), y_values, label=model_name, color=color[i],
                linestyle=line_styles[i], linewidth=3.5, marker=marker, markeredgewidth=3, markerfacecolor='white', markersize=8)
    
    # 상단 차트 설정
    #ax1.set_xlim(list(results.values())[0].index.min(), list(results.values())[0].index.max())
    ax1.set_xlim(-0.5, len(shared_index)-0.5)
    ax1.set_xticks(range(len(shared_index)))
    ax1.set_xticklabels([])
    ax1.set_ylim(ax1_ylim)
    ax1.set_yticks(np.arange(ax1_ylim[0], ax1_ylim[1]+0.1, 0.2))
    #ax1.set_yticks(np.arange(0, 1.1, 0.2))
    ax1.tick_params(axis='y', labelsize=18)
    
    # 70분위, 90분위 기준선 표시
    if perc90 is not None:
        ax1.axhline(perc90, color='red', linestyle='--', lw=2)
    if perc70 is not None:
        ax1.axhline(perc70, color='green', linestyle='--', lw=2)

    # 모델 범례 (왼쪽 상단)
    model_legend = ax1.legend(loc='upper left', frameon=False, fontsize=15)
    
    # 위험/주의 기준선을 위한 별도 범례 생성 (ET 범례 아래)
    threshold_lines = []
    threshold_labels = []
    
    if perc90 is not None:
        threshold_lines.append(plt.Line2D([0], [0], color='red', linestyle='--', lw=2))
        threshold_labels.append("위험")
    if perc70 is not None:
        threshold_lines.append(plt.Line2D([0], [0], color='green', linestyle='--', lw=2))
        threshold_labels.append("주의")
    
    if threshold_lines:
        threshold_legend = ax1.legend(threshold_lines, threshold_labels, 
                                    loc='upper left', bbox_to_anchor=(0.82, 1), frameon=False, fontsize=13)
        # 위험/주의 텍스트를 굵게 설정
        for i, text in enumerate(threshold_legend.get_texts()):
            text.set_weight('bold')
            if '위험' in text.get_text():
                text.set_color('red')
            elif '주의' in text.get_text():
                text.set_color('green')
        
        # 두 범례를 모두 표시하기 위해 첫 번째 범례를 다시 추가
        ax1.add_artist(model_legend)
    
    # 위기기간 표시
    if crises is not None:
        render_crises2(crises, ylim=[0, 1], ax=ax1, ispred=True, shared_index=shared_index)
    
    # 하단: 기여도 분해 차트
    #periods = impacts.index.get_level_values(0).unique()
    for period_idx, period in enumerate(periods):
        bottom_pos = 0
        bottom_neg = 0
        for feature_id, feature_group in zip(feature_ids, feature_groups):
            impact = impacts.loc[(period, feature_id)][0]
            if impact > 0:
                ax2.bar(period_idx, impact, bottom=bottom_pos, color=palette[feature_group], width=bar_width)
                bottom_pos += impact
            else:
                ax2.bar(period_idx, impact, bottom=bottom_neg, color=palette[feature_group], width=bar_width)
                bottom_neg += impact
    
    # 하단 차트 설정
    ax2.set_xlim([-0.5, len(periods) - 0.5])
    ax2.set_xticks(range(len(periods)))
    
    # 자동으로 시간 단위를 감지하여 라벨 형식 결정
    def get_time_labels(periods):
        """시간 간격에 따라 적절한 라벨 형식을 반환"""
        if len(periods) <= 1:
            return [period.strftime('%Y-%m-%d') for period in periods]
        
        # 시간 간격 계산 (첫 두 시점 기준)
        time_diff = (periods[1] - periods[0]).days
        
        if time_diff <= 7:  # 주별 또는 일별 데이터
            # 매 3번째마다 표시 (너무 많으면 생략)
            if len(periods) > 10:
                return [period.strftime('%m/%d') if i % 3 == 0 else '' 
                       for i, period in enumerate(periods)]
            else:
                return [period.strftime('%m/%d') for period in periods]
        elif time_diff <= 35:  # 월별 데이터 
            # 분기별 표시
            return [period.strftime('%b\n%y') if period.month in [3, 6, 9, 12] else '' 
                   for period in periods]
        else:  # 분기별 또는 연별 데이터
            return [period.strftime('%Y-%m') for period in periods]
    
    xticklabels = get_time_labels(periods)
    ax2.set_xticklabels(xticklabels, fontsize=14)
    
    # y축 설정
    y_top = impacts[impacts >= 0].groupby('period').sum().max()[0]
    y_bottom = impacts[impacts < 0].groupby('period').sum().min()[0]
    ax2.set_ylim([max(y_bottom*1.2, -1), min(y_top*1.2, 1)])
    #ax2.set_ylim([-0.3, 0.3])
    ax2.tick_params(axis='y', labelsize=18)
    
    # y=0 기준선 표시
    ax2.axhline(0, color='black', linewidth=1)
    
    # 범례 설정
    labels = np.array([label for label in palette]).reshape(legend_row, legend_col).T.flatten()
    handles = [plt.Rectangle((0, 0), 1, 1, color=palette[label]) for label in labels]
    ax2.legend(handles, labels, loc='lower center', frameon=False, ncol=legend_col,
               bbox_to_anchor=(0.5, -0.39), fontsize=11, columnspacing=0.5)
    
    # 그리드 설정
    ax2.grid(axis="both", linestyle='--', linewidth=0.9, alpha=0.9)
    
    if print_text:
        # 하단에 요인분해 결과 텍스트 추가
        def format_impact_text(impacts, periods, feature_ids, feature_names=None, top_n=5):
            """요인분해 결과를 텍스트로 포맷팅하는 헬퍼 함수"""
            text_blocks = []
            
            for period in periods[-6:]:  # 최근 6개월만 표시
                # 해당 시점의 모든 변수 영향도 추출
                try:
                    period_impacts = impacts.loc[period]
                    if hasattr(period_impacts, 'iloc') and len(period_impacts.shape) > 1:
                        period_impacts = period_impacts.iloc[:, 0]  # 첫 번째 컬럼 사용
                    
                    # 절댓값 기준으로 정렬하여 상위 n개 선택
                    top_impacts = period_impacts.reindex(
                        period_impacts.abs().sort_values(ascending=False).index
                    ).head(top_n)
                    
                    # 시점별 텍스트 생성 (한 줄로)
                    impact_items = []
                    for i, (feature_id, impact_value) in enumerate(top_impacts.items()):
                        # 변수명 매핑
                        if feature_names and feature_id in feature_ids:
                            try:
                                feature_idx = feature_ids.index(feature_id)
                                display_name = feature_names[feature_idx][:8]  # 이름 길이 단축
                            except:
                                display_name = str(feature_id)[:8]
                        else:
                            display_name = str(feature_id)[:8]
                        
                        # 영향도 방향 표시
                        direction = "↑" if impact_value > 0 else "↓"
                        impact_items.append(f"{direction}{display_name} ({abs(impact_value):.3f})")
                    
                    # 한 줄로 결합
                    period_text = f"{period.strftime('%Y-%m-%d')}: {', '.join(impact_items)}"

                    text_blocks.append(period_text)
                    
                except Exception as e:
                    print(f"Warning: Could not process period {period}: {e}")
                    continue
            
            return text_blocks

        # 메인 텍스트 출력 부분
        text_y_start = -0.06  # 텍스트 시작 위치 (범례 아래)

        # feature_names가 있는지 확인 (있다면 사용, 없다면 None)
        feature_names = None
        try:
            if 'feature_names' in locals() or hasattr(feature_ids, '__len__'):
                # feature_ids와 같은 길이의 이름이 있는지 확인
                pass  # 일단 None으로 유지
        except:
            pass

        # 텍스트 블록 생성
        text_blocks = format_impact_text(impacts, periods, feature_ids, feature_names, top_n=5)

        # 텍스트 배치 (세로로 배열)
        line_height = 0.025  # 줄 간격
        for idx, text_block in enumerate(text_blocks):
            text_y_position = text_y_start - idx * line_height
            
            # 텍스트 출력 (왼쪽 정렬)
            fig.text(0.05, text_y_position, text_block, 
                        fontsize=10, verticalalignment='top', fontfamily='Malgun Gothic', color='black', weight='bold') #

        # 전체 제목 추가
        fig.text(0.05, text_y_start + 0.02, "예측시점별 요인분해 결과 (상위 5개, 절대값 기준, ↑양수 ↓음수)", 
                    fontsize=12, ha='left', weight='bold', color='darkblue')

    plt.tight_layout()
    # 동적으로 하단 여백 조정 (시점 수에 따라)
    #bottom_margin = 0.2 + len(periods) * 0.02
    #plt.subplots_adjust(bottom=bottom_margin)
    #plt.show()
    
    if is_save_fig:
        fig.savefig(fig_name, dpi=300, bbox_inches='tight', transparent=True)


def plot_pred_decomp_all(results_dict, impacts_dict, feature_ids, feature_groups, 
                        crises_dict=None, perc70_dict=None, perc90_dict=None,
                        line_styles=['-', '--', '-.', ':'], palette=DEFAULT_PALETTE, 
                        bar_width=0.7, legend_row=1, legend_col=8, ax1_ylim=[0, 0.4], 
                        figsize=(20, 8), fig_name='pred_decomp_all', is_save_fig=True):
    """
    모든 부문(전체, 은행, 비은행)의 조기경보모형 예측결과와 변수별 기여도를 2x3 서브플롯에 표시한다.

    Args:
        results_dict: 부문별 예측결과 딕셔너리 {'전체': results, '은행': results_bank, '비은행': results_nbank}
        impacts_dict: 부문별 기여도 데이터 딕셔너리 {'전체': impacts, '은행': impacts_bank, '비은행': impacts_nbank}
        feature_ids: 입력 변수명 목록
        feature_groups: 입력 변수 그룹명 목록
        crises_dict: 부문별 위기기간 데이터 딕셔너리 (지정한 경우만 표시)
        perc70_dict: 부문별 70분위 임계치 딕셔너리
        perc90_dict: 부문별 90분위 임계치 딕셔너리
        line_styles: 모델이 여러개인 경우 차트를 구분하기 위한 라인 스타일
        palette: 변수 그룹별 색상
        bar_width: 단일 바 넓이
        legend_row: 범례 행 개수
        legend_col: 범례 열 개수
        ax1_ylim: 상단 차트 y축 범위
        figsize: figure 크기
        fig_name: 저장할 figure 파일명
        is_save_fig: 저장 여부
    """
    
    color = ['darkblue', 'grey', 'black', 'red', 'green', 'yellow', 'purple', 'orange', 'brown']
    
    # 2x3 서브플롯 생성 (상단 3개: 예측결과, 하단 3개: 기여도)
    fig, axes = plt.subplots(2, 3, figsize=figsize, 
                             gridspec_kw={'height_ratios': [3, 2], 'width_ratios': [1, 1, 1], 'wspace': 0.15, 'hspace': 0.15})
    
    # 부문별 설정
    sectors = ['전체', '은행', '비은행']
    sector_titles = ['전체', '은행', '비은행']
    
    for col, (sector, sector_title) in enumerate(zip(sectors, sector_titles)):
        if sector not in results_dict or sector not in impacts_dict:
            continue
            
        results = results_dict[sector]
        impacts = impacts_dict[sector]
        
        # 각 부문별 임계치 설정
        perc70 = perc70_dict.get(sector) if perc70_dict else None
        perc90 = perc90_dict.get(sector) if perc90_dict else None
        crises = crises_dict.get(sector) if crises_dict else None
        
        # 상단 차트 (예측결과)
        ax1 = axes[0, col]
        
        periods = impacts.index.get_level_values(0).unique()
        
        # periods가 비어있는 경우 예외처리
        if len(periods) == 0:
            ax1.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax1.transAxes)
            axes[1, col].text(0.5, 0.5, 'No data available', ha='center', va='center', transform=axes[1, col].transAxes)
            continue
        
        # 인덱스 타입에 따라 적절히 처리
        if not isinstance(periods[0], pd.Timestamp):
            try:
                shared_index = pd.to_datetime(sorted(set(periods).intersection(results[next(iter(results))].index)))
            except:
                shared_index = sorted(set(periods).intersection(results[next(iter(results))].index))
        else:
            shared_index = sorted(set(periods).intersection(results[next(iter(results))].index))
        
        # 예측결과 차트
        for i, (model_name, result) in enumerate(results.items()):
            marker = 'o' if 'ET' in model_name else None
            x_values = result.loc[shared_index].index
            y_values = result.loc[shared_index]["pred"]
            ax1.plot(range(len(x_values)), y_values, label='경보지수', color=color[i],
                    linestyle=line_styles[i], linewidth=3.5, marker=marker, 
                    markeredgewidth=3, markerfacecolor='white', markersize=8)
        
        # 상단 차트 설정
        ax1.set_xlim(-0.5, len(shared_index)-0.5)
        ax1.set_xticks(range(len(shared_index)))
        ax1.set_xticklabels([])
        ax1.set_ylim(ax1_ylim)
        #ax1.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
        ax1.set_yticks(np.arange(ax1_ylim[0], ax1_ylim[1]+0.1, 0.1))
        ax1.tick_params(axis='y', labelsize=16)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
        ax1.set_title(sector_title, fontsize=16, pad=5) # , fontweight='bold'
        ax1.grid(axis="y", linestyle='--', linewidth=0.5, alpha=0.7)

        # 임계치 라인 표시
        if perc90 is not None:
            ax1.axhline(perc90, color='red', linestyle='--', lw=2.5, alpha=0.8)
            ax1.text(len(shared_index)-1, perc90 + 0.002, f"위험({perc90:.2f})", 
                    color='red', fontsize=12, ha='right', va='bottom', fontweight='bold', alpha=0.8)
        if perc70 is not None:
            ax1.axhline(perc70, color='orange', linestyle='--', lw=2.5, alpha=0.8)
            ax1.text(len(shared_index)-1, perc70 + 0.002, f"주의({perc70:.2f})", 
                    color='orange', fontsize=12, ha='right', va='bottom', fontweight='bold', alpha=0.8)

        # 범례
        if col == 0:
            # 모델 범례
            #model_legend = ax1.legend(loc='upper left', frameon=False, fontsize=10)
            ax1.legend(ncol=1, loc=1, frameon=False, fontsize=15, bbox_to_anchor=(0.72, -0.889))
            
            # # 임계치 범례 (오른쪽 상단)
            # threshold_lines = []
            # threshold_labels = []
            
            # if perc90 is not None:
            #     threshold_lines.append(plt.Line2D([0], [0], color='red', linestyle='--', lw=2))
            #     threshold_labels.append("위험")
            # if perc70 is not None:
            #     threshold_lines.append(plt.Line2D([0], [0], color='orange', linestyle='--', lw=2))
            #     threshold_labels.append("주의")
            
            # if threshold_lines:
            #     threshold_legend = ax1.legend(threshold_lines, threshold_labels, 
            #                                 loc='upper right', frameon=False, fontsize=10)
            #     for text in threshold_legend.get_texts():
            #         text.set_weight('bold')
            #     ax1.add_artist(model_legend)
        
        # 위기기간 표시
        if crises is not None:
            render_crises2(crises, ylim=ax1_ylim, ax=ax1, ispred=True, shared_index=shared_index)
        
        # 하단 차트 (기여도 분해)
        ax2 = axes[1, col]
        
        for period_idx, period in enumerate(periods):
            bottom_pos = 0
            bottom_neg = 0
            for feature_id, feature_group in zip(feature_ids, feature_groups):
                impact = impacts.loc[(period, feature_id)][0]
                if impact > 0:
                    ax2.bar(period_idx, impact, bottom=bottom_pos, color=palette[feature_group], width=bar_width)
                    bottom_pos += impact
                else:
                    ax2.bar(period_idx, impact, bottom=bottom_neg, color=palette[feature_group], width=bar_width)
                    bottom_neg += impact
        
        # 하단 차트 설정
        ax2.set_xlim([-0.5, len(periods) - 0.5])
        ax2.set_xticks(range(len(periods)))
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
        
        # 시간 라벨 자동 설정
        def get_time_labels(periods):
            if len(periods) <= 1:
                return [period.strftime('%Y-%m-%d') for period in periods]
            
            time_diff = (periods[1] - periods[0]).days
            
            if time_diff <= 7:  # 주별 또는 일별 데이터
                if len(periods) > 10:
                    return [period.strftime('%m.%d') if i % 3 == 0 else '' 
                           for i, period in enumerate(periods)], period.strftime('%Y')
                else:
                    return [period.strftime('%m/%d') for period in periods]
            elif time_diff <= 35:  # 월별 데이터 
                return [period.strftime('%b\n%y') if period.month in [3, 6, 9, 12] else '' 
                       for period in periods]
            else:  # 분기별 또는 연별 데이터
                return [period.strftime('%Y-%m') for period in periods]
        
        xticklabels, year = get_time_labels(periods)
        ax2.set_xticklabels(xticklabels, fontsize=10)
        
        # y축 설정
        y_top = impacts[impacts >= 0].groupby('period').sum().max()[0]
        y_bottom = impacts[impacts < 0].groupby('period').sum().min()[0]
        #ax2.set_ylim([max(y_bottom*1.2, -1), min(y_top*1.2, 1)])
        margin = 0.05
        ax2.set_ylim(-0.1 - 0.05, 0.1 + 0.05)
        ax2.set_yticks([-0.1, 0, 0.1])
        ax2.tick_params(axis='y', labelsize=16)
        ax2.tick_params(axis='x', labelsize=16)
        
        # y=0 기준선 표시
        ax2.axhline(0, color='black', linewidth=1)
        
        # 범례 (가장 왼쪽 하단 차트에만)
        if col == 0:
            labels = np.array([label for label in palette]).reshape(legend_row, legend_col).T.flatten()
            handles = [plt.Rectangle((0, 0), 1, 1, color=palette[label]) for label in labels]
            ax2.legend(handles, labels, loc='lower center', frameon=False, ncol=legend_col,
                      bbox_to_anchor=(1.8, -0.4), fontsize=15, columnspacing=0.5)
        
        # 그리드 설정
        ax2.grid(axis="both", linestyle='--', linewidth=0.5, alpha=0.7)

        # 제목
        # main_title = '금융·외환시장 조기경보지수'
        # update_info = f'(업데이트: {year}.{xticklabels[-1]})'
        
        # fig.text(0.5, 1.01, main_title, fontsize=24, ha='center', va='top')
        # fig.text(0.5, 0.96, update_info, fontsize=15, ha='center', va='top', alpha=0.7)

    plt.tight_layout()
    
    if is_save_fig:
        fig.savefig(f'figure/{fig_name}.png', dpi=300, bbox_inches='tight')
    
    plt.show()
    #return fig


def plot_pdp_oneway(model, x, feature_names, top=15, n_cols=3, width=15, height=25):
    """partial dependency plot을 변수별로(one-way) 표시한다.

    Args:
        model: 조기경보모형
        x: 입력 변수
        feature_names: 입력 변수명 목록
        top: 중요도순으로 표시할 변수 개수
        n_cols: 차트 열 개수
        width: 차트 넓이
        height: 차트 높이
    """
    top_features = np.argsort(model.feature_importances_)[-top:]
    # 전체 pdp를 한번에 표시
    pdp = PartialDependenceDisplay.from_estimator(
        model, x, n_cols=n_cols, features=top_features, feature_names=feature_names,
        kind='both', ice_lines_kw={'color': 'gray', 'alpha': 0.3}, pd_line_kw={'color': 'red'})
    pdp.figure_.set_figwidth(width)
    pdp.figure_.set_figheight(height)
    # 개별 차트의 제목 설정 및 축라벨/범례 제거
    nrows = np.ceil(len(top_features) / n_cols).astype(int)
    for row in range(nrows):
        for col in range(n_cols):
            pdp.axes_[row][col].set_title(pdp.axes_[row][col].get_xlabel())
            pdp.axes_[row][col].set_ylim([0.0, 1.0])
            pdp.axes_[row][col].set_xlabel('')
            pdp.axes_[row][col].set_ylabel('')
            legend = pdp.axes_[row][col].get_legend()
            legend.set_visible(False)


def plot_pdp_twoway(model, x, feature_names, feature_types,
                    top=6, n_cols=2, width=16, height=30):
    """partial dependency plot을 변수의 쌍으로(two-way) 표시한다.

    Args:
        model: 조기경보모형
        x: 입력 변수
        feature_names: 입력 변수명 목록
        feature_types: 입력 변수 유형(취약성, 트리거) 목록
        top: 중요도순으로 표시할 변수 개수
        n_cols: 차트 열 개수
        width: 차트 넓이
        height: 차트 높이
    """
    # 각 그룹(취약성, 트리거)별 변수의 조합을 생성
    top_features = np.argsort(model.feature_importances_)[-top:]
    top_vul_features = [idx for idx in top_features if feature_types[idx]=='취약성']
    top_trg_features = [idx for idx in top_features if feature_types[idx]=='트리거']
    top_feature_combinations = [(vul, trg) for vul in top_vul_features for trg in top_trg_features]
    # 각 행/열별 subplot으로 pdp를 표시
    nrows = np.ceil(len(top_feature_combinations) / n_cols).astype(int)
    fig, axes = plt.subplots(nrows, n_cols, figsize=(width, height))
    for i, (vul_idx, trg_idx) in enumerate(top_feature_combinations):
        ax = axes[i//n_cols, i%n_cols] if nrows > 1 else axes[i%n_cols]
        PartialDependenceDisplay.from_estimator(
            model, x, grid_resolution=20, ax=ax,
            features=[(vul_idx, trg_idx)], feature_names=feature_names, kind='average')
        ax.set_title(f'(취약성) {feature_names[vul_idx]} & (트리거) {feature_names[trg_idx]}', fontsize=18)
        
def print_decomposed(et_impacts, period, top=5):
    ret = pd.to_numeric(et_impacts.loc[period]['impact'].droplevel(0), errors='coerce')
    print(ret.nlargest(top))
    
def map_fsi_thr_to_prob(fsi, fsi_warn_thr, fsi_risk_thr, prob):
    p_warn = fsi.le(fsi_warn_thr).mean()
    p_risk = fsi.le(fsi_risk_thr).mean()
    
    thr_warn = prob.quantile(p_warn)
    thr_risk = prob.quantile(p_risk)
    
    return thr_warn, thr_risk

def plot_cfpi_fsi(cfpi_fsi, figsize=(8, 4), save_path=None, dpi=300):
    """
    CFPI와 FSI 비교 플롯 함수
    
    Parameters:
    -----------
    cfpi_fsi : pandas.DataFrame
        CFPI와 FSI 데이터가 포함된 데이터프레임
    figsize : tuple, default=(8, 4)
        그래프 크기
    save_path : str, optional
        저장할 파일 경로 (None이면 저장하지 않음)
    dpi : int, default=300
        저장 시 해상도
    """
    
    # Create figure and primary axis
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Convert PeriodIndex to DatetimeIndex for better plotting
    cfpi_fsi_plot = cfpi_fsi.copy()
    if hasattr(cfpi_fsi_plot.index, 'to_timestamp'):
        cfpi_fsi_plot.index = cfpi_fsi_plot.index.to_timestamp()
    
    # Plot both CFPI indices on primary axis (left)
    ax1.plot(cfpi_fsi_plot.index, cfpi_fsi_plot['cfpi'], color='red', label='CFPI (확장, 좌축)')
    ax1.plot(cfpi_fsi_plot.index, cfpi_fsi_plot['cfpi_old'], color='blue', linestyle='--', label='CFPI (기존, 좌축)')
    
    # Create secondary axis (right) for fsi
    ax2 = ax1.twinx()
    ax2.plot(cfpi_fsi_plot.index, cfpi_fsi_plot['fsi_fin'], color='gray', label='FSI 금융시장(우축)', linewidth=1.5)
    
    # Add correlation information to title
    correlation = cfpi_fsi_plot['cfpi'].corr(cfpi_fsi_plot['fsi_fin'])
    plt.title(f'Comparison of CFPI and FSI, Correlation: {correlation:.2f}')
    
    # Create combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.grid(True)
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    plt.show()
    
    return fig, ax1, ax2

def get_vintage_date_range(end_date_str, months_back=3):
    """주어진 종료일로부터 지정된 개월 수만큼 이전부터의 빈티지 파일 날짜 범위를 계산"""
    end_date = pd.to_datetime(end_date_str)
    start_date = end_date - relativedelta(months=months_back)
    
    # 시작일을 해당 월의 첫째 주 월요일로 조정
    first_day_of_month = start_date.replace(day=1)
    days_to_monday = (0 - first_day_of_month.weekday()) % 7
    start_monday = first_day_of_month + pd.Timedelta(days=days_to_monday)
    
    return start_monday.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')

def get_vintage_files(vintage_folder, start_date, end_date):
    """지정된 날짜 범위에 해당하는 빈티지 파일 목록을 반환"""
    import os
    
    if not os.path.exists(vintage_folder):
        print(f"빈티지 폴더가 존재하지 않습니다: {vintage_folder}")
        return []
    
    all_files = [f for f in os.listdir(vintage_folder) if f.endswith('.csv')]
    
    # 날짜 형식의 파일만 필터링
    date_files = []
    for file in all_files:
        try:
            file_date = pd.to_datetime(file.replace('.csv', ''))
            if pd.to_datetime(start_date) <= file_date <= pd.to_datetime(end_date):
                date_files.append(file)
        except:
            continue
    
    # 날짜순으로 정렬
    date_files.sort()
    return date_files

def gen_vintage_combinded(vintage_folder, vintage_files, features):
    # 빈티지 데이터를 결합하여 X_vintage_combined 생성
    vintage_data_list = []

    for vintage_file in vintage_files:
        
        # 빈티지 데이터 로드
        vintage_path = os.path.join(vintage_folder, vintage_file)
        vintage_data = pd.read_csv(vintage_path, index_col=0)
        vintage_data.index = pd.to_datetime(vintage_data.index)
        
        # NaN 값을 이전값으로 채우기 (forward fill)
        vintage_data = vintage_data.ffill()
        vintage_data = vintage_data.fillna(0)
        
        # 해당 시점의 가장 최근 월말 데이터 선택
        vintage_date = vintage_file.replace('.csv', '')
        vintage_datetime = pd.to_datetime(vintage_date)
        
        if len(vintage_data) > 0:
            # 가장 최근 월말 데이터 선택
            latest_month_data = vintage_data.iloc[-1:].copy()
            
            # feature 추출
            X_vintage_single = latest_month_data[features['id']].copy()
            
            # 빈티지 날짜로 인덱스 설정
            X_vintage_single.index = [vintage_datetime]
            
            vintage_data_list.append(X_vintage_single)

    # 전체 빈티지 데이터 결합
    if vintage_data_list: # 빈티지 데이터가 있는 경우
        X_vintage_combined = pd.concat(vintage_data_list, axis=0)
        
        print(f"\n=== X_vintage_combined ===")
        print(f"크기: {X_vintage_combined.shape}")
        print(f"인덱스 타입: {type(X_vintage_combined.index[0])}")
        print(f"인덱스 범위: {X_vintage_combined.index[0]} ~ {X_vintage_combined.index[-1]}")
        print(f"feature 수: {X_vintage_combined.shape[1]}")
        
    return X_vintage_combined if vintage_data_list else None