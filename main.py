import pandas as pd
import numpy as np
import pickle
from sklearn import *
from matplotlib import style, pyplot

data = pd.read_csv('student-mat.csv', sep=';')
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
predict = "G3"

X = np.array(data.drop([predict], 1))
Y = np.array(data["G3"])
best = 0
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.1)
# for i in range(100):
#     x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.1)
#     linear = linear_model.LinearRegression()
#     linear.fit(x_train, y_train)
#     acc = linear.score(x_test, y_test)
    # if best < acc:
    #     best = acc
        # with open("studentmodel.pickle", "wb") as f:
        #     pickle.dump(linear, f)
# print(best)
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)
print(linear.score(x_test,y_test))
print('intercept', linear.intercept_)
print('coefficient - ', linear.coef_)
prediction = linear.predict(x_test)
for i in range(len(prediction)):
    print(prediction[i], x_test[i], y_test[i])
style.use("ggplot")
p = 'G1'
pyplot.scatter(data[p], data['G3'])
pyplot.xlabel(p)
pyplot.ylabel('Final Grade')
pyplot.show()