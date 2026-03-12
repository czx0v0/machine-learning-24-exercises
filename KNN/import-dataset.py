# Author:czx
# 2024年10月10日16时13分03秒
from sklearn.neighbors import KNeighborsClassifier
x = [[1,2,3,4],[5,4,7,8],[5,3,8,4],[1,2,8,9],[8,5,2,3]]
y = [1,2,3,4,5]

K=KNeighborsClassifier(n_neighbors=3)
K.fit(x,y) #使用x和y训练K近邻分类器
P= K.predict([[9,5,1,3]])
print(P)# 2

