from os import listdir
from os.path import isdir, join
import librosa
import random
import numpy as np
import matplotlib.pyplot as plt
import python_speech_features

#Dataset path and view possible targets(폴더명보여주는 코드)
dataset_path = 'C:/Users/jmj30/Dropbox/카메라 업로드/Documentation/2020/2020 2학기/project/창의자율과제/DNN프로젝트/data1'
for name in listdir(dataset_path):
    if isdir(join(dataset_path,name)):
        print(name)

#Create an all targets list(폴더명 리스트로 생성)
all_targets = [name for name in listdir(dataset_path) if isdir(join(dataset_path,name))]
print(all_targets)

#Leave off background noise set
all_targets.remove('_background_noise_')
all_targets.remove('heylumos')
print(all_targets)

#see how many files are in each
num_samples = 0
for target in all_targets:
    print(len(listdir(join(dataset_path,target))))
    num_samples += len(listdir(join(dataset_path,target)))
print('Total samples:', num_samples)

#Settings
target_list = all_targets #all_targets자리
feature_sets_file = 'all_targets_mfcc_sets.npz' #feature들 저장하는 파일
perc_keep_samples = 0.1 #추출하고 훈련할 때 샘플들을 얼마나 사용할것인지(다 사용하는 것이 1.0)
val_ratio = 0.1 #cross_validation 할 data 비율
test_ratio=0.1 #test 할 data 비율
sample_rate = 8000 #16000
num_mfcc = 16 #frame당 mfcc개수
len_mfcc= 16 #mfcc길이

#create list of filenames along with ground truth vector (y)
#파일이름들 절대경로로 가져옴 각각의 feature들 자동으로 추출할수 있게 해줌
filenames = []
y=[]
for index, target in enumerate(target_list):
    print(join(dataset_path, target))
    filenames.append(listdir(join(dataset_path, target)))
    y.append(np.ones(len(filenames[index]))*index) #각각의 절대 경로안에 있는 파일들에 같은 index부여해서 y array로 만듦

#check ground truth Y vector
print(y)
for item in y:
    print(len(item)) 

#flatten filename and y vectors
filenames = [item for sublist in filenames for item in sublist]
y = [item for sublist in y for item in sublist]

#associate filenames with true output and shuffle
#파일 안에 있는 데이터들을 섞어서 학습을 진행함
filenames_y = list(zip(filenames, y))
random.shuffle(filenames_y)
filenames, y = zip(*filenames_y)

#Only keep the specified number of samples(shorter extraction/training)
print(len(filenames))
filenames = filenames[:int(len(filenames)*perc_keep_samples)]
print(len(filenames))

#calculate validation and test set sizes
val_set_size = int(len(filenames)*val_ratio)
test_set_size = int(len(filenames)*test_ratio)

#break dataset apart into train, validation, and test sets
filenames_val = filenames[:val_set_size]
filenames_test = filenames[val_set_size:(val_set_size+test_set_size)]
filenames_train = filenames[(val_set_size+test_set_size):]

#break y apart into train, validation, and test sets
y_orig_val = y[:val_set_size]
y_orig_test = y[val_set_size:(val_set_size+test_set_size)]
y_orig_train = y[(val_set_size+test_set_size):]

#여기까지 하면 mfcc 추출준비 끝남

#Function: Create MFCC from give path
def calc_mfcc(path):

    #Load wavefile
    #path에서 wav 파일들 가져와서 위에 설정한 sample rate에 맞춰서 변환하는 작업
    signal, fs = librosa.load(path, sr=sample_rate)

    #Create MFCCs from sound clip
    #변환한 wave form에서 mfcc를 추출하는 작업
    mfccs = python_speech_features.base.mfcc(signal,
                                            samplerate=fs,
                                            winlen=0.256, #window size
                                            winstep=0.050, #windo 사이의 거리
                                            numcep=num_mfcc, #처음부터 12만큼의 mfcc가 필요(위에서 설정)
                                            nfilt=26, #default=26 filter
                                            nfft = 2048, #fft를 할때 사용하는 sample의 수는 윈도우 크기에 따름, 기본값은 512, 
                                            #참고 실험에서는 2048이었는데 WARNING:root:frame length (3200) is greater than FFT size (2048), frame will be truncated. Increase NFFT to avoid.
                                            #이 결과 값에 따라서 3200으로 변경
                                            preemph=0.0, #preemphasis filter
                                            ceplifter=0, #lifting opteration on final coefficient(more robust against noise)
                                            appendEnergy=False, #mfcc의 0번째 요소가 종종 버려지기 때문에 이건 그것을 그 프레임 안의 총 에너지를 대신하는 것으로 대신하는 것)
                                            winfunc=np.hanning) #window function: hamming/hanning window 는 FFT가 원하지 않는 요소들을 고주파에서 만들어 내는 것을 막는다
    return mfccs.transpose()

#TEST: Construct test set by computing MFCC of WAV file
#먼저 training data에서 500개 sample에 대해서만 mfcc를 추출(이 경우 144개)
# 각각의 파일들이 12개의 mfcc 모양을 보여줌
prob_cnt = 0
x_test = []
y_test = []
for index, filename in enumerate(filenames_train):
    #Stop after 500
    if index >= 500:
        break

    #create path from given filename and target item
    path = join(dataset_path, target_list[int(y_orig_train[index])],filename)

    #create MFCCs
    mfccs = calc_mfcc(path)

    if mfccs.shape[1] == len_mfcc:
        x_test.append(mfccs)
        y_test.append(y_orig_train[index])
    else:
        print("Dropped:", index, mfccs.shape)
        prob_cnt+=1

#위 결과를 확인했을 때 파일이 손상되었거나 1초가 넘지 않는다며 얼마만큼의 파일들이 그러는지 확인하는 코드
print('% of problematic samples:', prob_cnt/500) #% of problematic samples: 1.0

#TEST: test shorter MFCC
#!pip install playsound
from playsound import playsound

idx = 10

#Create path from given filename and target item
path = join(dataset_path, target_list[int(y_orig_train[idx])],filenames_train[idx])

#Create MFCCs
mfccs = calc_mfcc(path)
print("MFCCs:",mfccs)

#plot MFCC
fig = plt.figure()
plt.imshow(mfccs, cmap='inferno',origin='lower')

#TEST: Play problem sounds
print(target_list[int(y_orig_train[idx])])
playsound(path)

#dataset의 quality가 길이가 짧아서 안좋다면 앞뒤에 silence나 백색소음 붙여서 길이를 늘리는 방법과 그냥 그걸 뺴고 train할 수도 있다.
#문제가 있는 sample들 없애고 train 시키기
#Function: Create MFCCs, keeping only ones of desired length
def extract_features(in_files, in_y):
    prob_cnt = 0
    out_x = []
    out_y = []
    
    for index, filename in enumerate(in_files):
        #create path from given filename and target item
        path = join(dataset_path, target_list[int(in_y[index])],filename)
        #check to make sure we're reading a .wav file
        if not path.endswith('.wav'):
            continue
        #create MFCCs
        mfccs = calc_mfcc(path)

        #only keep MFCCs with given lenth
        if mfccs.shape[1] == len_mfcc:
            out_x.append(mfccs)
            out_y.append(in_y[index])
        else:
            print('dropped', index, mfccs.shape)
            prob_cnt += 1
    return out_x,out_y, prob_cnt

#create train, validation, and test sets
x_train, y_train, prob = extract_features(filenames_train,y_orig_train)
print('removed percentage:',prob/len(y_orig_train))
x_val, y_val, prob = extract_features(filenames_val,y_orig_val)
print('removed percentage:',prob/len(y_orig_val))
x_test, y_test, prob = extract_features(filenames_test,y_orig_test)
print('removed percentage:',prob/len(y_orig_test))

#Save features and truth vector(y) sets to disk
#위 결과에서 나온 손상된 파일들을 제거한 나머지 행렬들을 제일 위에 설정한 npz 파일에 넣어준다
#이는 feature들과 그에 따른 lable을 저장하는 과정   
np.savez(feature_sets_file,
        x_train = x_train,
        y_train = y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test)

#TEST: Load features
feature_sets = np.load(feature_sets_file)
print(feature_sets.files)

print(len(feature_sets['x_train']))
print(len(feature_sets['y_val']))