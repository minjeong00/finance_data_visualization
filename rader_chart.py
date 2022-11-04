import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from math import pi

# 코랩_ver
df = pd.read_excel("/content/기능별단체별세출결산2010.xlsx")

df_temp1 = df.drop(columns=['회계연도', '자치단체코드', '자치단체명', '분야코드', '부문명', '부문코드'])
df_field_pivot = pd.pivot_table(df_temp1, index=['지역명'], columns=['분야명'], values='세출결산액', aggfunc = 'sum')
df_field_pivot.reset_index(level=0, inplace=True)
df_field_pivot = df_field_pivot.drop(columns=['기타', '예비비'], axis=1)
df_field_pivot = df_field_pivot.fillna(0)

# 전체_스케일 조정 X
labels = df_field_pivot.columns[1:]
num_labels = len(labels)

angles = [x / float(num_labels) * (2 * pi) for x in range(num_labels)]  # 각 등분점
angles += angles[:1]  # 시작점으로 다시 돌아와야하므로 시작점 추가

my_palette = plt.cm.get_cmap("Set2", len(df_field_pivot.index))

fig = plt.figure(figsize=(20, 30))
fig.set_facecolor('white')

for i, row in df_field_pivot.iterrows():
    data = df_field_pivot.iloc[i].drop('지역명').tolist()
    data += data[:1]

    ax = plt.subplot(5, 4, i + 1, polar=True)
    ax.set_theta_offset(pi / 2)  ## 시작점
    ax.set_theta_direction(-1)  ## 그려지는 방향 시계방향

    plt.xticks(angles[:-1], labels, fontsize=13)  ## x축 눈금 라벨
    ax.tick_params(axis='x', which='major', pad=15)  ## x축과 눈금 사이에 여백을 준다.

    ax.set_rlabel_position(0)  ## y축 각도 설정(degree 단위)-
    # plt.yticks([0,2,4,6,8,10],['0','2','4','6','8','10'], fontsize=10) ## y축 눈금 설정
    # plt.ylim(0,10)

    ax.plot(angles, data, color='skyblue', linewidth=2, linestyle='solid')  ## 레이더 차트 출력
    ax.fill(angles, data, color='skyblue', alpha=0.4)  ## 도형 안쪽에 색을 채워준다.

    plt.title(row.지역명, size=20, color='skyblue', x=-0.2, y=1.2, ha='left')  ## 타이틀은 캐릭터 클래스로 한다.

plt.tight_layout(pad=1)  ## subplot간 패딩 조절
plt.show()

# 전체_스케일 조정 O
df_field_pivot1 = df_field_pivot[['공공질서및안전','과학기술','교육','국토및지역개발','농림해양수산','문화및관광','보건','사회복지','산업ㆍ중소기업','수송및교통','일반공공행정','환경보호']]
X = df_field_pivot1.values
my_scaler = MinMaxScaler()
X = my_scaler.fit_transform(X)

XX = df_field_pivot1.values
my_scaler2 = StandardScaler()
XX = my_scaler2.fit_transform(XX)

df_field_minmax = pd.DataFrame(data=X, columns=df_field_pivot1.columns)
df_field_standard = pd.DataFrame(data=XX, columns=df_field_pivot1.columns)

df_field_minmax['지역명'] = df_field_pivot[['지역명']].values
df_field_standard['지역명'] = df_field_pivot[['지역명']].values

from math import pi

labels = df_field_minmax.columns[:-1]
num_labels = len(labels)

angles = [x / float(num_labels) * (2 * pi) for x in range(num_labels)]  # 각 등분점
angles += angles[:1]  # 시작점으로 다시 돌아와야하므로 시작점 추가

my_palette = plt.cm.get_cmap("Set2", len(df_field_minmax.index))

fig = plt.figure(figsize=(20, 30))
fig.set_facecolor('white')

for i, row in df_field_minmax.iterrows():
    data = df_field_minmax.iloc[i].drop('지역명').tolist()
    data += data[:1]

    ax = plt.subplot(5, 4, i + 1, polar=True)
    ax.set_theta_offset(pi / 2)  ## 시작점
    ax.set_theta_direction(-1)  ## 그려지는 방향 시계방향

    plt.xticks(angles[:-1], labels, fontsize=13)  ## x축 눈금 라벨
    ax.tick_params(axis='x', which='major', pad=15)  ## x축과 눈금 사이에 여백을 준다.

    ax.set_rlabel_position(0)  ## y축 각도 설정(degree 단위)-
    # plt.yticks([0,2,4,6,8,10],['0','2','4','6','8','10'], fontsize=10) ## y축 눈금 설정
    # plt.ylim(0,10)

    ax.plot(angles, data, color='skyblue', linewidth=2, linestyle='solid')  ## 레이더 차트 출력
    ax.fill(angles, data, color='skyblue', alpha=0.4)  ## 도형 안쪽에 색을 채워준다.

    plt.title(row.지역명, size=20, color='skyblue', x=-0.2, y=1.2, ha='left')  ## 타이틀은 캐릭터 클래스로 한다.

plt.tight_layout(pad=1)  ## subplot간 패딩 조절
plt.show()