import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import RobustScaler

from sklearn.linear_model import Ridge
import sklearn

warnings.filterwarnings('ignore')

train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")
sample_submission = pd.read_csv("./data/sample_submission.csv")
plt.rc('font', family='NanumGothic')
plt.rc('axes', unicode_minus=False)

categorial_variable = ["임대건물구분", "지역", "공급유형", "자격유형"]
continious_variable = ["총세대수", "전용면적", "전용면적별세대수", "공가수", "임대보증금", "임대료",
                       "도보 10분거리 내 지하철역 수(환승노선 수 반영)", "도보 10분거리 내 버스정류장 수",
                       "단지내주차면수"]
target_variable = ["등록차량수"]

null_variable = ["임대보증금", "임대료",
                 "도보 10분거리 내 지하철역 수(환승노선 수 반영)", "도보 10분거리 내 버스정류장 수"]


# 임대보증금, 임대료, 도보 10분거리 내 지하철역 수(환승노선 수 반영), 도보 10분거리 내 버스정류장 수


def pre_processing(x):
    x = x.fillna(0)
    x.loc[x['임대료'] == '-', ['임대료']] = 0
    x.loc[x['임대보증금'] == '-', ['임대보증금']] = 0
    x[['임대료', '임대보증금']] = x[['임대료', '임대보증금']].astype('int64')
    x.isnull().sum()

    x = pd.get_dummies(x, columns=categorial_variable)
    scaler = RobustScaler()
    x[continious_variable] = scaler.fit_transform(x[continious_variable])
    return x


def get_removed_code(x):
    return x.drop('단지코드', axis=1)


df = train

df = pd.get_dummies(df, columns=categorial_variable)
differ_variables = ['공급유형_공공임대(5년)', '공급유형_공공임대(10년)', '자격유형_B', '자격유형_F', '자격유형_O',
                    '지역_서울특별시', '공급유형_공공분양', '공급유형_장기전세']

X_train = pre_processing(train.drop(['등록차량수'], axis=1))
y_train = df[['단지코드', '등록차량수']]
if len(test[test['자격유형'].isnull() is True]) > 0:
    test.loc[test['자격유형'].isnull() is True, ['자격유형']] = ('A', 'C')
X_test = pre_processing(test)
for c in differ_variables:
    X_test[c] = 0
id_code_train = train['단지코드']
id_code_test = test['단지코드']

print(X_train.columns)
print(X_test.columns)

rfr = RandomForestRegressor(n_estimators=200, max_depth=25, min_samples_leaf=1,
                            min_samples_split=4, random_state=93)
model = rfr

train_X, test_X, train_y, test_y = train_test_split(X_train, y_train, test_size=0.2, random_state=93)

grouped = test_y.groupby(test_y['단지코드'])
test_y_unique = grouped.agg(lambda x: x.value_counts().index[0])

model.fit(get_removed_code(train_X), get_removed_code(train_y))
pred = model.predict(get_removed_code(test_X))
test_result = pd.DataFrame(test_X['단지코드'])
test_result['등록차량수'] = pred
test_practice = test_result.groupby(test_result['단지코드']).agg(lambda x: x.value_counts().index[0])
mean_absolute_error(test_y_unique, test_practice)

params = {
    'n_estimators': [200],
    'max_depth': [25],
    'min_samples_leaf': [1],
    'min_samples_split': [4]
}

grid = GridSearchCV(rfr, param_grid=params, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid.fit(train_X.drop('단지코드', axis=1), train_y.drop('단지코드', axis=1))
print(grid.best_params_, grid.best_score_)

model.fit(X_train.drop('단지코드', axis=1), y_train.drop('단지코드', axis=1))
pred = model.predict(X_test.drop('단지코드', axis=1))
test_result = pd.DataFrame(X_test['단지코드'])
test_result['등록차량수'] = pred
group_test = test_result.groupby(test_result['단지코드'])
test_x_unique = group_test.agg(lambda x: x.value_counts().index[0])
# accuracy = mean_absolute_error(test_y_unique, test_x_unique)
# accuracy
result = pd.DataFrame(pd.unique(id_code_test))
result = result.merge(test_x_unique, left_on=0, right_on='단지코드')
result.columns = ['code', 'num']
result.to_csv('./result/result3_2.csv', index=False)
# test_x_unique
# gbr: 72.57647194273466
# rfr: 19.73668358714044
# {'max_depth': 12, 'min_samples_leaf': 8, 'min_samples_split': 16, 'n_estimators': 100} -62.07652539964124
