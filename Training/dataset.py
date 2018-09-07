import pandas as pd
import numpy as np
from os import listdir

def load_dataset(directory, suffix):
    print('Loading ' + suffix + '... ', end='')
    positions_hand = []
    orientations_hand = []

    hand_data = pd.read_csv(directory + '/out_fused_' + suffix + '.csv', dtype='float32')
    mocap_data = pd.read_csv(directory + '/out_mocap_' + suffix + '.csv', dtype='float32')
    
    print('Done.')
    return hand_data, mocap_data

def load_all(directory):
    files = listdir(directory)
    filtered = []
    for file in files:
        if file.startswith('out_'):
            file = file.replace('out_fused_', '')
            file = file.replace('out_mocap_', '')
            file = file.replace('.csv', '')
            filtered.append(file)
        
    filtered = list(set(filtered))
    hand_data_all, mocap_data_all = [], []
    for suffix in filtered:
        hand_data, mocap_data = load_dataset(directory, suffix)
        hand_data_all.append(hand_data)
        mocap_data_all.append(mocap_data)
    
    hand_data_all = pd.concat(hand_data_all)
    mocap_data_all = pd.concat(mocap_data_all)
    
    return hand_data_all, mocap_data_all

def extract_columns(dataset, names, suffix):
    columns = []
    for name in names:
        columns.append(name + '.' + suffix + '.x')
        columns.append(name + '.' + suffix + '.y')
        columns.append(name + '.' + suffix + '.z')
    output = dataset[columns].values
    return output.reshape(output.shape[0], -1, 3)

def split_dataset(X, Y, split_point):
    '''Split a dataset in a training set + validation set'''
    split_index = int(X.shape[0] * split_point)
    X_tr = X[:split_index]
    Y_tr = Y[:split_index]
    X_va = X[split_index:]
    Y_va = Y[split_index:]
    return X_tr, Y_tr, X_va, Y_va

def load_hand_template(filename, markers):
    data = pd.read_csv(filename)
    D = pd.DataFrame({'marker': markers})
    data = D.reset_index().merge(data).sort_values("index").drop("index", 1)
    return extract_columns(data, ['marker'], 'pos').reshape(-1, 3).astype('float32')

def pre_process(positions_hand, orientations_hand, positions_mocap):
    # Remove rows with NaNs
    indices = np.sum(positions_mocap != positions_mocap, axis=(1,2)) == 0
    positions_hand = positions_hand[indices]
    orientations_hand = orientations_hand[indices]
    positions_mocap = positions_mocap[indices]
    print('Hand:', positions_hand.shape, orientations_hand.shape)
    print('Mocap:', positions_mocap.shape)
    
    # Correct angles in order to normalize their ranges to [-1, 1] and center them around 0
    orientations_hand[orientations_hand > 180] = orientations_hand[orientations_hand > 180] - 360
    orientations_hand = orientations_hand / 180
    
    return positions_hand, orientations_hand, positions_mocap

def rigid_motion(X, Y):
    X_mean = np.mean(X, axis=0)
    Y_mean = np.mean(Y, axis=0)
    H = (Y - Y_mean).T.dot(X - X_mean)
    U, S, V = np.linalg.svd(H)
    R = V.T.dot(U.T)
    if np.linalg.det(R) < 0:
        V[2] = -V[2]
        R = V.T.dot(U.T)
    offset = -X_mean.dot(R) + Y_mean
    return R, offset