import numpy as np

def compute_ranges_numerical_features(feature_intervals, indices_categorical_features):
    mask_num = np.ones(len(feature_intervals), bool)
    mask_num[indices_categorical_features] = False
    num_feature_intervals = feature_intervals[mask_num]
    num_feature_ranges = np.array([x[1] - x[0] for x in num_feature_intervals])
    return num_feature_ranges


def get_mask_categorical_features(num_features, indices_categorical_features):
    mask_categorical = np.zeros(num_features, bool)
    if indices_categorical_features is not None and len(indices_categorical_features) > 0:
        mask_categorical[indices_categorical_features] = True
    return mask_categorical


def fix_categorical_features(genes, feature_intervals, indices_categorical_features):

    is_single_candidate = len(genes.shape) == 1
    if is_single_candidate:
        genes = genes.reshape((1,-1))

    for i in indices_categorical_features:

        categories_i = feature_intervals[i]

        distances = np.abs(np.repeat(genes[:,i], len(categories_i)).reshape((-1,len(categories_i))) - np.array(categories_i))
        closest_category_indices = np.argmin(distances, axis=1)
        closest_categories = categories_i[closest_category_indices]
        genes[:,i] = closest_categories

    if is_single_candidate:
        genes = genes[0]

    return genes


def fix_features(genes, x, feature_intervals, 
    indices_categorical_features=None, plausibility_constraints=None):

    is_single_candidate = len(genes.shape) == 1
    if is_single_candidate:
        genes = genes.reshape((1,-1))

    # fix plausibilities
    for i, plau_c in enumerate(plausibility_constraints):
      # the value cannot change (e.g., not actionable feature)
      if plau_c == '=':
        genes[:,i] = x[i]
      # the value can only increase (e.g, age)
      elif plau_c == '>=':
        genes[:,i] = np.where(genes[:,i] > x[i], genes[:,i], x[i])
      # the value can only decrease
      elif plau_c == '<=':
        genes[:,i] = np.where(genes[:,i] < x[i], genes[:,i], x[i])

    # fix categorical features_to_use
    genes = fix_categorical_features(genes, feature_intervals, indices_categorical_features)
    if is_single_candidate:
        genes = genes.reshape((1,-1))

    # fix bounds
    mask_categorical = get_mask_categorical_features(len(x), indices_categorical_features)
    for i, is_c in enumerate(mask_categorical):
        if is_c:
            continue

        low = feature_intervals[i][0]
        high = feature_intervals[i][1]

        genes[:,i] = np.where(genes[:,i] > low, genes[:,i], low)
        genes[:,i] = np.where(genes[:,i] < high, genes[:,i], high)
        
    if is_single_candidate:
        genes = genes[0]
    
    return genes