import numpy as np
import pandas as pd
import csv

data_dir = '../datasets/UCI_data/raw/'
processed_data_dir = '../datasets/UCI_data/preprocessed/'

# IRIS
data_fn = 'iris.data'
print (data_fn)

data = np.loadtxt(data_dir+data_fn, dtype='str', delimiter=',')
features = data[:,:-1].astype(np.float)
print (features.shape)
n = features.shape[0]

np.savetxt(processed_data_dir+'iris.features', features, delimiter=',', fmt='%.1f')
classes = np.zeros(n)
class_names = np.unique(data[:,-1])

print (len(class_names))
for j in range(len(class_names)):
    classes[data[:,-1] == class_names[j]] = j

np.savetxt(processed_data_dir+'iris.classes', classes, fmt='%d')

# ADULT
data_fn = 'adult.data'
print (data_fn)

data = np.loadtxt(data_dir+data_fn, dtype='str', delimiter=',')
features = data[:,:-1]
print (features.shape)
n = features.shape[0]
df = pd.DataFrame(features)
cols = [1, 3, 5, 6, 7, 8, 9, 10, 13]
for col in cols:
    df[col] = pd.Categorical(df[col], categories=df[col].unique()).codes

np.savetxt(processed_data_dir+'adult.features', df, delimiter=',', fmt='%s')
classes = np.zeros(n)
class_names = np.unique(data[:,-1])
print (len(class_names))

for j in range(len(class_names)):
    classes[data[:,-1] == class_names[j]] = j

np.savetxt(processed_data_dir+'adult.classes', classes, fmt='%d')

# WINE
data_fn = 'wine.data'
print (data_fn)

delim = ','

data = np.loadtxt(data_dir+data_fn, dtype='str', delimiter=delim)
features = data[:,1:]
print (features.shape)
n = features.shape[0]

np.savetxt(processed_data_dir+'wine.features', features, delimiter=',', fmt='%s')
classes = np.zeros(n)

class_names = np.unique(data[:,0])
print (len(class_names))
for j in range(len(class_names)):
    classes[data[:,0] == class_names[j]] = j

np.savetxt(processed_data_dir+'wine.classes', classes, fmt='%d')

# CAR
data_fn = 'car.data'
print (data_fn)

data = np.loadtxt(data_dir+data_fn, dtype='str', delimiter=',')
features = data[:,:-1]
print (features.shape)
n = features.shape[0]
df = pd.DataFrame(features)
cols = range(6)
for col in cols:
    df[col] = pd.Categorical(df[col], categories=df[col].unique()).codes

np.savetxt(processed_data_dir+'car.features', df, delimiter=',', fmt='%s')
classes = np.zeros(n)
class_names = np.unique(data[:,-1])
print (len(class_names))

for j in range(len(class_names)):
    classes[data[:,-1] == class_names[j]] = j

np.savetxt(processed_data_dir+'car.classes', classes, fmt='%d')

# BREAST-CANCER
data_fn = 'breast-cancer.data'
print (data_fn)

delim = ','

data = np.loadtxt(data_dir+data_fn, dtype='str', delimiter=delim)
features = data[:,2:]
print (features.shape)
n = features.shape[0]

np.savetxt(processed_data_dir+'breast-cancer.features', features, delimiter=',', fmt='%s')
classes = np.zeros(n)

class_names = np.unique(data[:,1])
print (len(class_names))
for j in range(len(class_names)):
    classes[data[:,1] == class_names[j]] = j

np.savetxt(processed_data_dir+'breast-cancer.classes', classes, fmt='%d')

# ABALONE
data_fn = 'abalone.data'
print (data_fn)

data = np.loadtxt(data_dir+data_fn, dtype='str', delimiter=',')
features = data[:,:-1]
print (features.shape)
n = features.shape[0]
df = pd.DataFrame(features)
df[0] = pd.Categorical(df[0], categories=df[0].unique()).codes

np.savetxt(processed_data_dir+'abalone.features', df, delimiter=',', fmt='%s')
classes = np.zeros(n)
class_names = np.unique(data[:,-1])
print (len(class_names))

for j in range(len(class_names)):
    classes[data[:,-1] == class_names[j]] = j

np.savetxt(processed_data_dir+'abalone.classes', classes, fmt='%d')


# WINE-QUALITY
data_fn = 'winequality.data'
print (data_fn)

data = np.loadtxt(data_dir+data_fn, dtype='str', delimiter=';', skiprows=1)
features = data[:,:-1]
print (features.shape)
n = features.shape[0]

np.savetxt(processed_data_dir+'winequality.features', features, delimiter=',', fmt='%s')
classes = np.zeros(n)
class_names = np.unique(data[:,-1])
print (len(class_names))

for j in range(len(class_names)):
    classes[data[:,-1] == class_names[j]] = j

np.savetxt(processed_data_dir+'winequality.classes', classes, fmt='%d')

# HEART-DISEASE
data_fn = 'heart-disease.data'
print (data_fn)

data = np.loadtxt(data_dir+data_fn, delimiter=',', dtype='str')
data[data == "?"] = "0"
features = data[:,:-1]
print (features.shape)
n = features.shape[0]

np.savetxt(processed_data_dir+'heart-disease.features', features, delimiter=',', fmt='%s')
classes = np.zeros(n)

class_names = range(2)
print (len(class_names))
classes = 1*(data[:, -1].astype(np.float) > 0)

np.savetxt(processed_data_dir+'heart-disease.classes', classes, fmt='%d')

# YEAST
data_fn = 'yeast.data'
print (data_fn)

data = np.loadtxt(data_dir+data_fn, dtype='str')
features = data[:,1:-1]
print (features.shape)
n = features.shape[0]

np.savetxt(processed_data_dir+'yeast.features', features, delimiter=',', fmt='%s')
classes = np.zeros(n)

class_names = np.unique(data[:,-1])
print (len(class_names))
for j in range(len(class_names)):
    classes[data[:,-1] == class_names[j]] = j

np.savetxt(processed_data_dir+'yeast.classes', classes, fmt='%d')

# INTERNET-ADS
data_fn = 'internet-ads.data'
print (data_fn)

data = []
with open(data_dir+data_fn, 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        data.append(row)

data = np.array(data)
data = np.char.strip(data)
data[data == "?"] = "0"
features = data[:, :-1]
print (features.shape)
n = features.shape[0]

np.savetxt(processed_data_dir+'internet-ads.features', features, delimiter=',', fmt='%s')
classes = np.zeros(n)

class_names = np.unique(data[:, -1])
print (len(class_names))
for j in range(len(class_names)):
    classes[data[:, -1] == class_names[j]] = j

np.savetxt(processed_data_dir+'internet-ads.classes', classes, fmt='%d')

# POKER
# unclear if we're looking at the same dataset.
