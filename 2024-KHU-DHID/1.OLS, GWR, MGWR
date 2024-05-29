import pandas as pd
import numpy as np

# 데이터 준비
file_path = 'D:/*/'

bd = pd.read_csv(file_path + '서울특별시_원룸 및 오피스텔 현황_20240229.csv', encoding='cp949')
bd['SGG_NM'] = bd['대지위치'].apply(lambda x: x.split()[1])
bd = bd.groupby('SGG_NM').agg({'호수': 'sum'}).reset_index()

hs = pd.read_csv(file_path + '서울특별시_자치구 법정동별 에너지사용량 통계_주거용_20230512.csv', encoding='cp949')
hs['SGG_NM'] = hs['자치구명']
hs = hs.groupby('SGG_NM').agg({'2022년 1월':'sum'}).reset_index()

en = pd.read_csv(file_path + '서울시 병원 인허가 정보.csv', encoding='cp949')
en['SGG_NM'] = en['도로명주소'].apply(lambda x: x.split()[1] if isinstance(x, str) else np.nan)
en.groupby('SGG_NM').agg({'2022년 1월':'sum'}).reset_index()
en = en.groupby('SGG_NM').size().reset_index(name='개수')

merged_df = bd.merge(hs, on='SGG_NM').merge(en, on='SGG_NM')

# OLS
import statsmodels.api as sm

# 독립변수(X)와 종속변수(y) 설정
X = merged_df[['개수', '호수']]
y = merged_df['2022년 1월']

# 상수항 추가 (절편)
X = sm.add_constant(X)

# OLS 모델 피팅
model = sm.OLS(y, X).fit()

# 결과 출력
print(model.summary())


# 랜덤포레스트
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 데이터 불러오기
#data = pd.read_csv('your_dataset.csv') # 데이터셋 파일 경로에 맞게 수정해주세요
data = merged_df

# 설명 변수와 종속 변수 선택
X = data[['개수', '호수']]
y = data['2022년 1월']

# 학습용과 테스트용으로 데이터 나누기
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 랜덤 포레스트 모델 생성
rf_model = RandomForestRegressor(random_state=42)

# 모델 학습
rf_model.fit(X_train, y_train)

# 테스트 데이터에 대한 예측
y_pred = rf_model.predict(X_test)

# 평가 지표 계산 (예를 들어, 평균 제곱 오차)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
