from random import shuffle
import glob
from VideoGenerator import DataGenerator

from keras import backend as K

##############################################################################
##############################################################################
#                           READ NTU DATASET
##############################################################################
##############################################################################

# Set path of NTU videos 
shuffle_data = True
path = 'C://NTUdataset'
orig_path = path + '//*.avi'  #You can use the original .AVI videos or pre-processed videos

# read addresses and labels
addr = sorted(glob.glob(orig_path))


# get index in the form of '007'
def get_index(train_index, test_index):
    for ind in range(len(train_index)):
        index = str(train_index[ind])
        if len(index) == 1:
            train_index[ind] = '00' + index
        elif len(index) == 2:
            train_index[ind] = '0' + index
    for ind in range(len(test_index)):
        index = str(test_index[ind])
        if len(index) == 1:
            test_index[ind] = '00' + index
        elif len(index) == 2:
            test_index[ind] = '0' + index

    return train_index, test_index


# define model
mode = 'cs'

# separate addresses according to the selected mode
train_origin = []
test_origin = []

if mode == 'cs':  # Cross_subject evaluation
    train_index = [1, 2, 4, 5, 8, 9, 13, 14, 15,
                   16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
    test_index = [3, 6, 7, 10, 11, 12, 20, 21, 22, 23, 24, 26, 29, 30, 32, 33, 36, 37, 39, 40]
    train_index, test_index = get_index(train_index, test_index)
    # Cross_Subject
    for dir in addr:
        person = dir[len(path) + 10:len(path) + 13]
        if person in train_index:
            train_origin.append(dir)
        elif person in test_index:
            test_origin.append(dir)
else:  # Cross_view evaluation
    train_index = [2, 3]
    test_index = [1]
    train_index, test_index = get_index(train_index, test_index)
    # Cross_Subject
    for dir in addr:
        person = dir[len(path) + 6:len(path) + 9]
        if person in train_index:
            train_origin.append(dir)
        elif person in test_index:
            test_origin.append(dir)

# to shuffle data
if shuffle_data:
    shuffle(train_origin)
    shuffle(test_origin)

# Parameters for data generation
dim = 224     # Desired final dimension (frames x dim x dim x 3)
frames = 32   # Desired number of frames (re-sampling)
params = {'dim': dim,
          'batch_size': 12,
          'n_channels': 3,
          'frames': frames,
          'path': path,
          'classes': 60,
          'shuffle': True}

# Create dictionaries
data_dict = {}
data_dict["train"] = train_origin
data_dict["validation"] = test_origin
# data_dict["test"]=test_origin

# Generatorsy
training_generator = DataGenerator(data_dict['train'], **params)
validation_generator = DataGenerator(data_dict['validation'], **params)
