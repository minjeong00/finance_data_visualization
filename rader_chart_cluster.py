# # 강원/경남/경북/전남/전북/충남
# fig, ax = plt.subplots(figsize=(4,4), subplot_kw=dict(polar=True))
# area_names = ['강원','경남','경북','전남','전북','충남']
#
# for area_name in area_names:
#   for i in range(len(df_field_minmax)):
#     if df_field_minmax.iloc[i,12] == area_name:
#       categories = list(df_field_minmax.columns[:-1])
#       money = list(df_field_minmax.iloc[i,[0,1,2,3,4,5,6,7,8,9,10,11]].values)
#
#       angles = [n/float(len(categories))*2*pi for n in range(len(categories))]
#       angles += angles[:1]
#       money += money[:1]
#
#       ax.set_rlabel_position(30)
#
#       ax.plot(angles, money, linewidth=1, linestyle='solid')
#       ax.fill(angles, money, alpha=0.4)
#       ax.set_title
#
#   plt.xticks(angles[:-1], categories, color='grey', size=12)
#   plt.title('강원/경남/경북/전남/전북/충남', size=16, color='blue',x=-0.2, y=1.2, ha='left')