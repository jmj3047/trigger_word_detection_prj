from os import listdir
from os.path import isdir, join
import tensorflow
from tensorflow.keras import layers, models
import numpy as np

#Create list of all targets(minus background noise)
dataset_path = 'C:/Users/jmj30/Dropbox/카메라 업로드/Documentation/2020/2020 2학기/project/창의자율과제/DNN프로젝트/data1'
all_targets = all_targets = [name for name in listdir(dataset_path) if isdir(join(dataset_path, name))]
all_targets.remove('_background_noise_')
all_targets.remove('heylumos')
print(all_targets)

#Settings
feature_sets_path = 'C:/Users/jmj30/Dropbox/카메라 업로드/Documentation/2020/2020 2학기/project/창의자율과제/DNN프로젝트'
#위에서 추출한 피처들 저장할 path
feature_sets_filename = 'all_targets_mfcc_sets.npz' 
model_filename = 'wake_word_stop_model.h5' #Neural network file 저장할 파일
wake_word = 'stop' #실험에 사용할 wakeword: 위에 all_targets 리스트에 저장되어 있는 것들 중 하나여야 함

#Load feature sets
feature_sets = np.load(join(feature_sets_path, feature_sets_filename))
print(feature_sets.files)

# Assign feature sets
x_train = feature_sets['x_train']
y_train = feature_sets['y_train']
x_val = feature_sets['x_val']
y_val = feature_sets['y_val']
x_test = feature_sets['x_test']
y_test = feature_sets['y_test']

#Look at tensor dimensions
print(x_train.shape)
print(y_val.shape)
print(x_test.shape)
#(a,b,c)의 형태로 출력됨: 여기서 a는 set안에 있는 샘플의 개수, b는 coefficients의 개수, c는 샘플당 coefficient의 집합 개수

#Peek at lables
print(y_val)
#이걸 프린트 하면 모든 데이터 폴더의 레이블이 나오는데 binary decision에서는 필요하지 않음
#그래서 타겟워드의 index만 안다면 numpy를 사용해 그거만 살리고 나머지는 동일하게 변경 한다. 

import sys
print(sys.version)