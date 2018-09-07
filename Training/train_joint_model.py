import numpy as np
from config import *
from dataset import *
from exporter import save_model

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation
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

print('Training on', subjects_train)
if perform_validation:
	print('Validating on', subjects_valid)
	hand_data_va = []
	mocap_data_va = []
	for subject in subjects_valid:
		hand_data_, mocap_data_ = load_all('data/' + subject)
		hand_data_va.append(hand_data_)
		mocap_data_va.append(mocap_data_)
	hand_data_va = pd.concat(hand_data_va)
	mocap_data_va = pd.concat(mocap_data_va)
	
	positions_mocap_va = extract_columns(mocap_data_va, all_markers, 'pos')
	positions_hand_va = extract_columns(hand_data_va, hand_features, 'pos')
	orientations_hand_va = extract_columns(hand_data_va, hand_features, 'rot')


print('---Dataset---')
print('Hand:', positions_hand.shape, orientations_hand.shape)
print('Mocap:', positions_mocap.shape)

positions_hand, orientations_hand, positions_mocap = pre_process(
    positions_hand, orientations_hand, positions_mocap)

if perform_validation:	
	positions_hand_va, orientations_hand_va, positions_mocap_va = pre_process(
		positions_hand_va, orientations_hand_va, positions_mocap_va)

hand = load_hand_template('data/hand.csv', all_markers)
print('Hand Template:', hand.shape)

def normalize_positions(positions_mocap):
    '''Put each mocap sample in object space'''
    error_count = 0
    # Compute rigid motion
    Y = hand[alignment_points_markers]
    for i in range(positions_mocap.shape[0]):
        X = positions_mocap[i, alignment_points_markers]
        try:
            R, offset = rigid_motion(X, Y)
        except:
            error_count += 1
        positions_mocap[i] = positions_mocap[i].dot(R) + offset
    print('Error count:', error_count)
    return positions_mocap

def extract_input_features(mocap_sample):
    '''Flatten the inout, e.g. a 9x3 matrix becomes a vector of length 27'''
    return mocap_sample.flatten()

def extract_output_features(hand_sample):
    '''Extract the angles of each joint'''
    flist = []
    for i in range(hand_sample.shape[0]):
        flist.append(hand_sample[i, features_components[i]])
    flist = [item for sublist in flist for item in sublist]
    return np.array(flist)

positions_mocap = normalize_positions(positions_mocap)
X_tr = np.array([extract_input_features(e) for e in positions_mocap])
Y_tr = np.array([extract_output_features(e) for e in orientations_hand])
print(X_tr.shape)
print(Y_tr.shape)

if perform_validation:
	positions_mocap_va = normalize_positions(positions_mocap_va)
	X_va = np.array([extract_input_features(e) for e in positions_mocap_va])
	Y_va = np.array([extract_output_features(e) for e in orientations_hand_va])
	va_pair = (X_va, Y_va)
	print(X_va.shape)
	print(Y_va.shape)
else:
	va_pair = None

# Neural network definition
model = Sequential()
model.add(Dense(200, activation='relu', input_shape=X_tr.shape[1:]))
model.add(Dropout(0.1))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(200, activation='relu'))
model.add(Dense(Y_tr.shape[1], activation='linear'))

opt = Adam(lr=0.001)

# The learning rate is halved if the training error has not improved over the last 5 epochs
lr_callback = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5,
                                verbose=1, mode='auto', epsilon=1e-6, cooldown=0, min_lr=0)

# Stops the training process upon convergence
stop_callback = EarlyStopping(monitor='loss', min_delta=1e-6, patience=11, verbose=1, mode='auto')

model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])
model.summary()

try:
    model.fit(X_tr, Y_tr, batch_size=32, epochs=1000, verbose=2,
          validation_data=va_pair,
          callbacks=[lr_callback, stop_callback])
except KeyboardInterrupt:
        # Do not throw away the model in case the user stops the training process
        pass

# Save the weights of the neural network to file
save_model(model, 'joint_model.bin')

if not perform_validation:
	print('***** Evaluating loss on the test set *****')
	print('Testing on', subjects_test)
	hand_data_te = []
	mocap_data_te = []
	for subject in subjects_test:
		hand_data_, mocap_data_ = load_all('data/' + subject)
		hand_data_te.append(hand_data_)
		mocap_data_te.append(mocap_data_)
	hand_data_te = pd.concat(hand_data_te)
	mocap_data_te = pd.concat(mocap_data_te)

	positions_mocap_te = extract_columns(mocap_data_te, all_markers, 'pos')
	positions_hand_te = extract_columns(hand_data_te, hand_features, 'pos')
	orientations_hand_te = extract_columns(hand_data_te, hand_features, 'rot')

	print('--- Test set ---')
	print('Hand:', positions_hand_te.shape, orientations_hand_te.shape)
	print('Mocap:', positions_mocap_te.shape)

	positions_hand_te, orientations_hand_te, positions_mocap_te = pre_process(
		positions_hand_te, orientations_hand_te, positions_mocap_te)

	positions_mocap_te = normalize_positions(positions_mocap_te)
	X_te = np.array([extract_input_features(e) for e in positions_mocap_te])
	Y_te = np.array([extract_output_features(e) for e in orientations_hand_te])
	print(X_te.shape)
	print(Y_te.shape)

	te_score = model.evaluate(X_te, Y_te, batch_size=16, verbose=0)
	print('Test set RMSE (degrees):', np.sqrt(te_score[0])*180)
	print('Test set MAE (degrees):', te_score[1]*180)
