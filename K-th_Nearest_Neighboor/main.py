import pandas as pd
import sklearn
from sklearn import model_selection, preprocessing
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv('car.data')

le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
persons = le.fit_transform(list(data["persons"]))
door = le.fit_transform(list(data["door"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

X = list(zip(maint, door, persons, safety, lug_boot))
y = list(cls)

x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1)
model = KNeighborsClassifier(n_neighbors=9)
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
predict = model.predict(x_test)
print(acc)
names = ["pes", "orta", "govy", "bet"]
for i in range(len(predict)):
    print("predicted:", names[predict[i]], "input", x_test[i], "actual", names[y_test[i]])
    n = model.kneighbors([x_test[i]], 9, True)
    print(n)
