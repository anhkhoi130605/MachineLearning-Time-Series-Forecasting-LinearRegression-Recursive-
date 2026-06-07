import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def create_ts_data(data,window_size=5):
    i=1
    while i<=window_size:
        data["co2_{}".format(i)] = data["co2"].shift(-i)
        i+=1
    data["target"] = data["co2"].shift(-i)
    data= data.dropna(axis=0)
    return data
data = pd.read_csv(r"D:\Data-Science+Machine-Learning\Practice-4(Times Series)\co2.csv")
data["time"] = pd.to_datetime(data["time"])
data["co2"] =data["co2"].interpolate()
data=create_ts_data(data)
# print(data.info())
# fig, ax = plt.subplots()
# ax.plot(data["time"], data["co2"])
# ax.set(xlabel="time", ylabel="co2")
# plt.show()
x=data.drop(["target","time"],axis=1)
y=data["target"]
train_ratio = 0.8
num_samples = len(x)
x_train = x[:int(num_samples*train_ratio)]
y_train = y[:int(num_samples*train_ratio)]
x_test = x[int(num_samples*train_ratio):]
y_test = y[int(num_samples*train_ratio):]
reg =LinearRegression()
reg.fit(x_train,y_train)
y_predict = reg.predict(x_test)
print("Mean Absolute Error:{} ".format(mean_absolute_error(y_test,y_predict)))
print("Mean Squared Error: {}".format(mean_squared_error(y_test,y_predict)))
print("R2 Score: {}".format(r2_score(y_test,y_predict)))
# Mean Absolute Error:0.36016447939447016
# Mean Squared Error: 0.2164843025903659
# R2 Score: 0.9909169586732987
fig, ax = plt.subplots()
ax.plot(data["time"][:int(num_samples*train_ratio)], data["co2"][:int(num_samples*train_ratio)],label="Train")
ax.plot(data["time"][int(num_samples*train_ratio):], data["co2"][int(num_samples*train_ratio):],label="Test")
ax.plot(data["time"][int(num_samples*train_ratio):], y_predict,label="Prediction")
ax.legend()
ax.grid()
ax.set(xlabel="time", ylabel="co2")
plt.show()
#Dự đoán đệ quy đa bước