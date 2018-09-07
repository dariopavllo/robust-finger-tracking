# If perform_validation is True, the model will be trained on subjects_train
# and validated againts subjects_valid. If false, it will be trained on
# subjects_train plus subjects_valid, and tested against subjects_test.
perform_validation = False
subjects_train = ['S1', 'S2']
subjects_valid = ['S3']
subjects_test = ['S4']


# The list of all markers (the order is important)
all_markers = ["LEHAR", "LEHAL", "WHARHAR", "LRIF", "LBAF", "LMIF", "LINF", "LPAL", "LTHF"]

# The list of alignment markers
alignment_points_markers = ["LEHAR", "LEHAL", "WHARHAR"]

# The joints whose angles must be predicted
hand_features = ["l_index1", "l_index2", "l_index3", "l_middle1", "l_middle2", "l_middle3",
    "l_pinky1", "l_pinky2", "l_pinky3", "l_ring1", "l_ring2", "l_ring3", "l_thumb1", "l_thumb2", "l_thumb3"]

# Degrees of freedom of each joint, where 0 = x, 1 = y, 2 = z.
# The order must correspond to the joints in "hand_features"
features_components = [
                  [0, 1, 2], [1], [1],
                  [0, 1, 2], [1], [1],
                  [0, 1, 2], [1], [1],
                  [0, 1, 2], [1], [1],
                  [0, 1, 2], [0, 1], [0]]


# Do not edit this part (convert string to index)
for i in range(len(alignment_points_markers)):
    alignment_points_markers[i] = all_markers.index(alignment_points_markers[i])