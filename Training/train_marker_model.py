import numpy as np
from config import *
from dataset import *
from exporter import save_model

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Ensure reproducibility of results
np.random.seed(1)

if not perform_validation:
	print('Validation mode disabled')
	subjects_train += subjects_valid
else:
	print('Validation mode enabled')

# Load training set
hand_data = []
mocap_data = []
for subject in subjects_train:
	hand_data_, mocap_data_ = load_all('data/' + subject)
	hand_data.append(hand_data_)
	mocap_data.append(mocap_data_)
hand_data = pd.concat(hand_data)
mocap_data = pd.concat(mocap_data)

positions_mocap = extract_columns(mocap_data, all_markers, 'pos')
positions_hand = extract_columns(hand_data, hand_features, 'pos')
orientations_hand = extract_columns(hand_data, hand_features, 'rot')

print('---Dataset---')
print('Hand:', positions_hand.shape, orientations_hand.shape)
print('Mocap:', positions_mocap.shape)

positions_hand, orientations_hand, positions_mocap = pre_process(
    positions_hand, orientations_hand, positions_mocap)

hand = load_hand_template('data/hand.csv', all_markers)
print('Hand Template:', hand.shape)


positions_hand, orientations_hand, positions_mocap = pre_process(
    positions_hand, orientations_hand, positions_mocap)

# If enabled, leave out a portion of the training set to validate the model
positions_mocap_tr = positions_mocap

# Neural network definition
model = Sequential()
model.add(Dense(200, activation='relu', input_shape=(len(all_markers) * 3,)))
model.add(Dense(150, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(len(all_markers) * 3, activation='linear'))

# The learning rate is halved if the training error has not improved over the last 5 epochs
lr_callback = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5,
                                verbose=1, mode='auto', epsilon=1e-7, cooldown=0, min_lr=0)

# Stops the training process upon convergence
stop_callback = EarlyStopping(monitor='loss', min_delta=1e-7, patience=11, verbose=1, mode='auto')

batch_size = 32

model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001), metrics=['mae'])
model.summary()

def extract_features(mocap_sample, occlusions):
    '''Given a motion capture sample, puts all markers in object space and sets missing values to 0.
    
    Arguments:
    mocap_sample -- the sample to process
    occlusions -- the set (indices) of occluded markers
    
    Returns: input and output of the neural network
    '''
    occluded = set(occlusions)
    available = set(np.arange(mocap_sample.shape[0])) - occluded
    available, occluded = list(sorted(available)), list(sorted(occluded))
    
    R, offset = rigid_motion(mocap_sample[available], hand[available])
    y = mocap_sample.dot(R) + offset
    x = y.copy()
    x[occluded] = 0

    return x.flatten(), y.flatten()


def generate_minibatch(X, min_occlusions, max_occlusions):
    '''Procedure for real-time minibatch creation and dataset augmentation.
    This runs in a parallel thread while the model is being trained.
    
    Arguments:
    X -- the training set
    min_occlusions -- the minimum number of markers occluded at once (-1 = default)
    max_occlusions -- the maximum numbers of markers occluded at once (-1 = default)
    '''
    counter = X.shape[0]
    while True:
        # Generate one minibatch
        X_batch = np.empty((batch_size, len(all_markers)*3))
        Y_batch = np.empty((batch_size, len(all_markers)*3))
        for i in range(batch_size):
            if counter == X.shape[0]:
                indices = np.random.permutation(X.shape[0])
                counter = 0
            # Select a random sample
            idx = indices[counter]
            counter += 1
            
            if min_occlusions != -1:
                num_occlusions = np.random.randint(min_occlusions, max_occlusions + 1)
            else:
                num_occlusions = np.random.choice(4, p=[0.7,  0.17,  0.08,  0.05]) + 1
            occluded_idxs = np.random.choice(len(all_markers), num_occlusions, replace=False)
            x, y = extract_features(X[idx], occluded_idxs)

            X_batch[i] = x
            Y_batch[i] = y

        yield (X_batch, Y_batch)
        
def evaluate_model(name, X, min_occlusions, max_occlusions):
    validation_steps = X.shape[0]//batch_size
    va_score = model.evaluate_generator(
                generate_minibatch(X, min_occlusions, max_occlusions),
                steps=validation_steps)
    print('--------',name,'--------')
    print('RMSE (centimeters):', np.sqrt(va_score[0])*100)
    print('MAE (centimeters):', va_score[1]*100)
        
        
try:
    model.fit_generator(generate_minibatch(positions_mocap_tr, -1, -1),
                    steps_per_epoch=positions_mocap_tr.shape[0]//batch_size,
                    epochs=1000,
                    verbose=2,
                    #validation_data=generate_minibatch(positions_mocap_va, 1, 3),
                    #validation_steps=positions_mocap_va.shape[0]//batch_size,
                    callbacks=[lr_callback, stop_callback])
except KeyboardInterrupt:
        # Do not throw away the model in case the user stops the training process
        pass
    
# Save the weights of the neural network to file
save_model(model, 'marker_model.bin')

# Perform validation (if enabled)
if perform_validation:
	print('***** Performing validation *****')
	test = subjects_valid
else:
	print('***** Evaluating loss on the test set *****')
	test = subjects_test
	
hand_data_te = []
mocap_data_te = []
for subject in test:
	hand_data_, mocap_data_ = load_all('data/' + subject)
	hand_data_te.append(hand_data_)
	mocap_data_te.append(mocap_data_)
hand_data_te = pd.concat(hand_data_te)
mocap_data_te = pd.concat(mocap_data_te)

positions_mocap_te = extract_columns(mocap_data_te, all_markers, 'pos')
positions_hand_te = extract_columns(hand_data_te, hand_features, 'pos')
orientations_hand_te = extract_columns(hand_data_te, hand_features, 'rot')
if not perform_validation:
	print('--- Test set ---')
else:
	print('--- Validation set ---')
	
print('Hand:', positions_hand_te.shape, orientations_hand_te.shape)
print('Mocap:', positions_mocap_te.shape)

positions_hand_te, orientations_hand_te, positions_mocap_te = pre_process(
    positions_hand_te, orientations_hand_te, positions_mocap_te)

evaluate_model('Random occlusions (1-4)', positions_mocap_te, -1, -1)
evaluate_model('0 occlusions', positions_mocap_te, 0, 0)
evaluate_model('1 occlusions', positions_mocap_te, 1, 1)
evaluate_model('2 occlusions', positions_mocap_te, 2, 2)
evaluate_model('3 occlusions', positions_mocap_te, 3, 3)
evaluate_model('4 occlusions', positions_mocap_te, 4, 4)