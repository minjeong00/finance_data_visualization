import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# 코랩_ver
df = pd.read_excel("/content/drive/MyDrive/Colab Notebooks/dataset/재정데이터시각화/2019_기능별단체별세출결산.xlsx")

df_temp1 = df.drop(columns=['회계연도', '자치단체코드', '자치단체명', '분야코드', '부문명', '부문코드'])
df_field_pivot = pd.pivot_table(df_temp1, index=['지역명'], columns=['분야명'], values='세출결산액', aggfunc = 'sum')
df_field_pivot.reset_index(level=0, inplace=True)
df_field_pivot = df_field_pivot.drop(columns=['기타', '예비비'], axis=1)
df_field_pivot = df_field_pivot.fillna(0)

df_final = df_field_pivot[ ['공공질서및안전', '과학기술', '교육',
          '국토및지역개발', '농림해양수산', '문화및관광', '보건','사회복지','산업ㆍ중소기업',''
                                                              '수송및교통','일반공공행정','환경보호']]

# 컬럼의 표준화를 실행한다.
X = df_final.values
my_scaler = MinMaxScaler()
X = my_scaler.fit_transform(X)
pd.DataFrame(data=X, columns=df_final.columns)

# 군집의 수 정하기
my_km = KMeans(n_clusters = 5 , random_state = 1)
my_km.fit(X)
my_centroids = my_km.cluster_centers_               # 개개 군집의 중심점.
my_cluster_labels = my_km.labels_                   # 군집 label.

# 주성분 분석 (PCA)를 활용한 차원축소 (2차원).
my_pca = PCA(n_components = 2)
transformed_comps = my_pca.fit_transform(X)
df_transformed_comps = pd.DataFrame(data = transformed_comps, columns = ['PC1', 'PC2'])
df_transformed_comps=df_transformed_comps.join(pd.Series(my_cluster_labels, name='cluster_label'))
df_transformed_comps

# 산점도 시각화.
my_colors = {0:'pink',1:'blue',2:'green',3:'yellow',4:'brown'}
my_names = {0: '충북/충남/강원/전남/전북/경남/경북', 1: '경기.', 2: '울산/대전/대구', 3:'서울',4:'제주/세종/광주/인천/부산'}

plt.figure(figsize = (6,6))
for a_cluster_n, df_small in df_transformed_comps.groupby('cluster_label'):
    plt.scatter(df_small['PC1'], df_small['PC2'], c = my_colors[a_cluster_n], label = my_names[a_cluster_n], s=50, alpha=0.6, marker="o" )

for i in range(len(X)):
  plt.text(transformed_comps[i, 0], transformed_comps[i, 1], str(df_field_pivot['지역명'][i]), fontdict={'weight':'bold', 'size':9} )

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Clusters')
plt.legend(loc="best")
plt.show()

# 5개가 좋을듯.
ssw = []
cluster_ns = range(2,14)
for n in cluster_ns:
    my_cluster = KMeans(n)
    my_cluster.fit(X)
    ssw.append(my_cluster.inertia_)

plt.figure(figsize = (6,6))
plt.plot(cluster_ns, ssw)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Sum of Squares Within')
plt.show()

# 주성분 값을 히트맵으로 시각화
plt.matshow(my_pca.components_, cmap='viridis')
plt.colorbar()
plt.yticks([0,1],['첫 번째 주성분','두 번째 주성분'])
plt.xticks(range(len(df_final.columns)), df_final.columns, rotation=90)
plt.show()


