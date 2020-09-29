import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")

# kullanılacak dataların başlıklarını yazıp gerisi siliyoruz
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
# yeni şekli ile datanın durumu print(data.head())

# predict bizim sonucunu görmek istediğimiz verinin etiket adı
predict = "G3"
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])
# bu adımdaki fonksiyon arrayi 4 ayırıyor x ve y eğitimi ve test işlemi içinde 0.1 yani yüzde 10 test için ayırıyor
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
"""""
best = 0
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
    
    linear = linear_model.LinearRegression()
    #train deki değerlere baglı olarak bir en uygun dogru seciyor
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print(acc)
    if acc > best:
        acc = best
        with open("StudentModel.pickle","wb") as f:
            pickle.dump(linear, f)"""

pickle_in = open("StudentModel.pickle", "rb")
linear = pickle.load(pickle_in)

print('Co: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

prediction = linear.predict(x_test)

for a in range(len(prediction)):
    print(prediction[a], x_test[a], y_test[a])
p = 'failures'
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()
