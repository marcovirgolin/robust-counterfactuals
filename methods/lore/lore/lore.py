from lore import pyyadt
import random

from lore.neighbor_generator import *
from lore.gpdatagenerator import calculate_feature_values
from lore.util import compute_ranges_numerical_features 
from lore.distance_functions import np_gower_distance



def get_cfe(idx_record2explain, X2E, dataset, blackbox,
            ng_function=genetic_neighborhood, #generate_random_data, #genetic_neighborhood, random_neighborhood
            discrete_use_probabilities=False,
            continuous_function_estimation=False,
            desired_class = 0,
            returns_infos=False, path='./methods/lore/lore/', sep=';', log=False):
            
    random.seed(0)
    class_name = dataset['class_name']
    columns = dataset['columns']
    discrete = dataset['discrete']
    continuous = dataset['continuous']

    # Dataset Preprocessing
    dataset['feature_values'] = calculate_feature_values(X2E, columns, class_name, discrete, continuous, 1000,
                                                         discrete_use_probabilities, continuous_function_estimation)

    dfZ, x = dataframe2explain(X2E, dataset, idx_record2explain, blackbox)

    # Generate Neighborhood
    dfZ, Z = ng_function(dfZ, x, blackbox, dataset)

    # Get those of desired class
    dfZ_to_consider = dfZ[dfZ['LABEL'] == desired_class]
    del dfZ_to_consider['LABEL']
    Ztc = dfZ_to_consider.to_numpy()

    # get the best one
    feature_intervals = dataset['feature_intervals']
    indices_categorical_features = dataset['indices_categorical_features']
    num_feature_ranges = compute_ranges_numerical_features(feature_intervals, indices_categorical_features)

    z_best = None
    smallest_dist = np.inf
    for z in Ztc:
        d = np_gower_distance(z, x, num_feature_ranges, indices_categorical_features)
        if d < smallest_dist:
            smallest_dist = d
            z_best = z

    return z_best

def explain(idx_record2explain, X2E, dataset, blackbox,
            ng_function=genetic_neighborhood, #generate_random_data, #genetic_neighborhood, random_neighborhood
            discrete_use_probabilities=False,
            continuous_function_estimation=False,
            returns_infos=False, path='./methods/lore/lore/', sep=';', log=False):

    random.seed(0)
    class_name = dataset['class_name']
    columns = dataset['columns']
    discrete = dataset['discrete']
    continuous = dataset['continuous']
    features_type = dataset['features_type']
    label_encoder = dataset['label_encoder']
    possible_outcomes = dataset['possible_outcomes']

    # Dataset Preprocessing
    dataset['feature_values'] = calculate_feature_values(X2E, columns, class_name, discrete, continuous, 1000,
                                                         discrete_use_probabilities, continuous_function_estimation)

    dfZ, x = dataframe2explain(X2E, dataset, idx_record2explain, blackbox)

    # Generate Neighborhood
    dfZ, Z = ng_function(dfZ, x, blackbox, dataset)

    # Build Decision Tree
    dt, dt_dot = pyyadt.fit(dfZ, class_name, columns, features_type, discrete, continuous,
                            filename=dataset['name'], path=path, sep=sep, log=log)

    # Apply Black Box and Decision Tree on instance to explain
    bb_outcome = blackbox.predict(x.reshape(1, -1))[0]

    dfx = build_df2explain(blackbox, x.reshape(1, -1), dataset).to_dict('records')[0]
    cc_outcome, rule, tree_path = pyyadt.predict_rule(dt, dfx, class_name, features_type, discrete, continuous)

    # Apply Black Box and Decision Tree on neighborhood
    y_pred_bb = blackbox.predict(Z)
    y_pred_cc, leaf_nodes = pyyadt.predict(dt, dfZ.to_dict('records'), class_name, features_type,
                                           discrete, continuous)

    def predict(X):
        y, ln, = pyyadt.predict(dt, X, class_name, features_type, discrete, continuous)
        return y, ln

    # Update labels if necessary
    if class_name in label_encoder:
        cc_outcome = label_encoder[class_name].transform(np.array([cc_outcome]))[0]  

    if class_name in label_encoder:
        y_pred_cc = label_encoder[class_name].transform(y_pred_cc)

    # Extract Coutnerfactuals
    diff_outcome = get_diff_outcome(bb_outcome, possible_outcomes)
    counterfactuals = pyyadt.get_counterfactuals(dt, tree_path, rule, diff_outcome,
                                                 class_name, continuous, features_type)

    explanation = (rule, counterfactuals)

    infos = {
        'bb_outcome': bb_outcome,
        'cc_outcome': cc_outcome,
        'y_pred_bb': y_pred_bb,
        'y_pred_cc': y_pred_cc,
        'dfZ': dfZ,
        'Z': Z,
        'dt': dt,
        'tree_path': tree_path,
        'leaf_nodes': leaf_nodes,
        'diff_outcome': diff_outcome,
        'predict': predict,
    }

    if returns_infos:
        return explanation, infos

    return explanation


def is_satisfied(x, rule, discrete, features_type):
    for col, val in rule.items():
        if col in discrete:
            if str(x[col]).strip() != val:
                return False
        else:
            if '<=' in val and '<' in val and val.find('<=') < val.find('<'):
                val = val.split(col)
                thr1 = pyyadt.yadt_value2type(val[0].replace('<=', ''), col, features_type)
                thr2 = pyyadt.yadt_value2type(val[1].replace('<', ''), col, features_type)
                # if thr2 < x[col] <= thr1: ok
                if x[col] > thr1 or x[col] <= thr2:
                    return False
            elif '<' in val and '<=' in val and val.find('<') < val.find('<='):
                val = val.split(col)
                thr1 = pyyadt.yadt_value2type(val[0].replace('<', ''), col, features_type)
                thr2 = pyyadt.yadt_value2type(val[1].replace('<=', ''), col, features_type)
                # if thr2 < x[col] <= thr1: ok
                if x[col] >= thr1 or x[col] < thr2:
                    return False
            elif '<=' in val:
                thr = pyyadt.yadt_value2type(val.replace('<=', ''), col, features_type)
                if x[col] > thr:
                    return False
            elif '>' in val:
                thr = pyyadt.yadt_value2type(val.replace('>', ''), col, features_type)
                if x[col] <= thr:
                    return False
    return True


def get_covered(rule, X, dataset):
    covered_indexes = list()
    for i, x in enumerate(X):
        if is_satisfied(x, rule, dataset['discrete'], dataset['features_type']):
            covered_indexes.append(i)
    return covered_indexes


def apply_rule(rule, z, x, feature_names, eps=0.0):

    for feature_name in list(rule.keys()):
        idx_feature = [i for i, fn in enumerate(feature_names) if fn == feature_name][0]
        rule_value = rule[feature_name]

        op_found = None
        feat_value = None

        # first check for a simple '>=','>','<=','<','==','='
        for op in ['>=','>','<=','<','==','=']: # note: order matters
            if rule_value.startswith(op):
                op_found = op
                feat_value = float(rule_value.replace(op,''))
                break
            else:
                # equality can be expressed without = or ==
                try:
                    feat_value = float(rule_value)
                    op_found = '='
                    break
                except ValueError:
                    feat_value = None
                    op_found = None


        if op_found is not None:
            if op_found == '==' or op_found == '=':
                z[idx_feature] = feat_value
            elif op_found == '>' or op_found == '>=':
                z[idx_feature] = feat_value + eps
            elif op_found == '<' or op_found == '<=':
                z[idx_feature] = feat_value - eps
            continue

        # The checks before failed, now check for stuff like "0.3 < X <= 0.7"
        if '<' in rule_value and '<=' in rule_value:
            idx_left = rule_value.find('<')
            idx_right = rule_value.find('<=')

            if idx_right < idx_left:
                temp = idx_right
                idx_right = idx_left
                idx_left = temp

            num_left = float(rule_value[:idx_left])
            num_right = float(rule_value[idx_right+2:])
            # get closest to x
            dist_left = np.abs(x[idx_feature] - (num_left+eps))
            dist_right = np.abs(x[idx_feature] - (num_right-eps))
            if dist_left < dist_right:
                z[idx_feature] = num_left + eps
            else:
                z[idx_feature] = num_right - eps
            continue


        # nothing worked
        raise ValueError("Could not recognize comparison operator in rule",rule)

    return z 


def explanation_to_cfe(explanation, x, feature_names, blackbox, desired_class):
    z = x.copy()

    rules = explanation[1]

    # found no rules
    if len(rules) == 0:
        return z

    # let's take the rule that influences less features
    if type(rules) == list and len(rules) > 1:
        sorted(rules, key=lambda r: len(r))
        rule = rules[0]
    else:
        rule = rules

    # repeat applying the rule until the decision boundary is reached
    # (since sometimes this does not happen with +eps)
    n_max_attempts = 15
    n_attempts = 0
    z = apply_rule(rule, z, x, feature_names, eps=0)
    
    eps=1e-3
    while blackbox.predict([z])[0] != desired_class and n_attempts < n_max_attempts: 
        z = apply_rule(rule, z, x, feature_names, eps=eps)
        eps *= 2
        n_attempts += 1
        
    return z