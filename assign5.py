from sklearn import datasets
from sklearn import ensemble, linear_model, tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

diabetes = datasets.load_diabetes()

x = diabetes.data
y = diabetes.target

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=169)


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=169)


model = ensemble.RandomForestRegressor(n_estimators = 23, max_depth=4, min_samples_leaf=15)
model.fit(x_train, y_train)
y_predict = model.predict(x_test)

ran_abs = mean_absolute_error(y_test, y_predict)
ran_sqr = mean_squared_error(y_test, y_predict)
ran_r2 = r2_score(y_test, y_predict)

print("Random Forest")
print (ran_abs, ran_sqr, ran_r2)

model1 = linear_model.LinearRegression()
model1.fit(x_train, y_train)
y_predict1 = model1.predict(x_test)

lin_abs = mean_absolute_error(y_test, y_predict1)
lin_sqr = mean_squared_error(y_test, y_predict1)
lin_r2 = r2_score(y_test, y_predict1)

print("Linear Regression")
print(lin_abs, lin_sqr, lin_r2)

model2 = tree.DecisionTreeRegressor(max_depth=4, min_samples_leaf=15)
model2.fit(x_train, y_train)
y_predict2 = model2.predict(x_test)

dec_tree_abs = mean_absolute_error(y_test, y_predict2)
dec_tree_sqr = mean_squared_error(y_test, y_predict2)
dec_tree_r2 = r2_score(y_test, y_predict2)

print("Decision Tree")
print(dec_tree_abs, dec_tree_sqr, dec_tree_r2)

