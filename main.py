from tkinter import *
from tkinter import filedialog
from tkinter.ttk import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from matplotlib import figure

csvfile_add = ""


def diagopenbox():
    global csvfile_add
    root.filename = filedialog.askopenfilename(initialdir="/", title="Select file",
                                               filetypes=(("csv files", "*.csv"), ("all files", "*.*")))
    csvfile_add = root.filename
    label_csv_file.config(text=csvfile_add)
    return csvfile_add


def ml_train(entry_index_value, entry_time_scaler_value,
             entry_dropout_value, entry_nurons_value, entry_epoch_value, entry_batch_size_value):
    label_training_status.config(text="Initiated")
    dataset_train = pd.read_csv(csvfile_add)
    var1 = int(list(dataset_train.columns.values).index(str(entry_index_value)))
    training_set = dataset_train.iloc[:, var1:var1 + 1].values
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)
    X_train = []
    y_train = []
    time_scaler = int(entry_time_scaler_value)
    for i in range(time_scaler, len(dataset_train)):
        X_train.append(training_set_scaled[i - time_scaler:i, 0])
        y_train.append(training_set_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    # Initialising the RNN
    regressor = Sequential()

    # Adding the first LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = int(entry_nurons_value), return_sequences=True, input_shape=(X_train.shape[1], 1)))
    regressor.add(Dropout(float(entry_dropout_value)))

    # Adding a second LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units=int(entry_nurons_value), return_sequences=True))
    regressor.add(Dropout(float(entry_dropout_value)))

    # Adding a third LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units=int(entry_nurons_value), return_sequences=True))
    regressor.add(Dropout( float(entry_dropout_value)))

    # Adding a fourth LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units=int(entry_nurons_value)))
    regressor.add(Dropout(float(entry_dropout_value)))

    # Adding the output layer
    regressor.add(Dense(units=1))
    label_training_status.config(text="Processing")
    # Compiling the RNN
    regressor.compile(optimizer='adam', loss='mean_squared_error')

    # Fitting the RNN to the Training set
    regressor.fit(X_train, y_train, epochs=int(entry_epoch_value), batch_size=int(entry_batch_size_value))
    label_training_status.config(text="Competed")
    # Svae the trained model
    regressor.save('model.h5')


def plotter(entry_index_value,time_scaler):
    regressor = load_model('model.h5')
    result = [0.0] * time_scaler
    dataset_train = pd.read_csv(csvfile_add)
    var1 = int(list(dataset_train.columns.values).index(str(entry_index_value)))
    training_set = dataset_train.iloc[:, var1:var1 + 1].values
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)
    var2 = int(list(dataset_train.columns.values).index(str(entry_index_value)))
    stock_price = dataset_train.iloc[:len(dataset_train) - time_scaler, var2:var2 + 1].values
    stock_price = stock_price.reshape(-1, 1)
    stock_price = sc.transform(stock_price)
    arr = []
    X_test = []
    for i in range(len(stock_price) - time_scaler, len(stock_price)):
        arr.append(stock_price[i:len(stock_price), 0])
    arr = pd.DataFrame(arr)
    arr = arr.fillna(0)
    arr = np.array(arr)
    for y in range(0, time_scaler - 1):
        X_test = arr
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        predicted_stock_price = regressor.predict(X_test)
        predicted_stock_price = sc.inverse_transform(predicted_stock_price)
        arr[y + 1][time_scaler - 1 - y] = predicted_stock_price[0][0]
    print("Done")
    plt.figure(figsize=(15, 9))
    plt.plot(predicted_stock_price, color='blue', label='Predicted')
    plt.legend()
    plt.show()





def ml_train_setup():
    entry_index_value = entry_index.get()
    entry_time_scaler_value = entry_time_scaler.get()
    entry_dropout_value = entry_dropout.get()
    entry_nurons_value = entry_nurons.get()
    entry_epoch_value = entry_epoch.get()
    entry_batch_size_value = entry_batch_size.get()
    ml_train(entry_index_value, entry_time_scaler_value, entry_dropout_value,
             entry_nurons_value, entry_epoch_value, entry_batch_size_value)


root = Tk()
#root.state('zoomed')
# menus
menu = Menu(root)
root.config(menu=menu)
subMenu = Menu(menu)
menu.add_cascade(label="test", menu=subMenu)
subMenu.add_command(label="text....")

# button to select csv file path
button_csv_file = Button(root, text="Select the\nCsv File", command=lambda: diagopenbox())
button_csv_file.grid(row=0, column=0)
label_csv_file = Label(root, text="")
label_csv_file.grid(row=0, column=1)

# Index String to be trained
label_index = Label(root, text='Enter The Column Index')
label_index.grid(row=1, column=0)
entry_index = Entry(root)
entry_index.grid(row=1, column=1)

# time_scaler
label_time_scaler = Label(root, text='Enter the time scaler')
label_time_scaler.grid(row=3, column=0)
entry_time_scaler = Entry(root)
entry_time_scaler.grid(row=3, column=1)
# length of test data
label_test_data = Label(root, text='Size Of test data')
label_test_data.grid(row=4, column=0)
entry_test_data = Entry(root)
entry_test_data.grid(row=4, column=1)
# dropout percent
label_dropout = Label(root, text='Dropout (0 -1)')
label_dropout.grid(row=5, column=0)
entry_dropout = Entry(root)
entry_dropout.grid(row=5, column=1)
# number of nurons
label_nurons = Label(root, text='Nurons')
label_nurons.grid(row=6, column=0)
entry_nurons = Entry(root)
entry_nurons.grid(row=6, column=1)
# epochs
label_epoch = Label(root, text='Epoch')
label_epoch.grid(row=7, column=0)
entry_epoch = Entry(root)
entry_epoch.grid(row=7, column=1)
# batch size
label_batch_size = Label(root, text='Batch Size')
label_batch_size.grid(row=8, column=0)
entry_batch_size = Entry(root)
entry_batch_size.grid(row=8, column=1)
# Start Training
label_train = Label(root, text='Batch Size')
label_train.grid(row=9, column=0)
button_train = Button(root, text='Start Training', command=lambda: ml_train_setup())
button_train.grid(row=9, column=1)
# Progress bar
#progress_bar = Progressbar(root, orient=HORIZONTAL, length=400, mode='determinate')
#progress_bar.grid(row=10, columnspan=2)

#Training Status
label_training_status_text = Label(root, text='Training Stats')
label_training_status_text.grid(row=10,column=0)
label_training_status = Label(root, text="Not Initiated")
label_training_status.grid(row=10, column=1)
#Plotter
label_plotter = Label(root,text='Plotter')
label_plotter.grid(row=11,column = 0)
buttom_plotter = Button(root,text='Plot',command = lambda:plotter(entry_index.get(),int(entry_time_scaler.get())))
buttom_plotter.grid(row=11,column=1)

root.mainloop()
