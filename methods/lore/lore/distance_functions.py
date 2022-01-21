import numpy as np


def normalized_euclidean_distance(x, y):
    return 0.5 * np.var(x - y) / (np.var(x) + np.var(y))


def simple_match_distance(x, y):
    count = 0
    for xi, yi in zip(x, y):
        if xi == yi:
            count += 1
    sim_ratio = 1.0 * count / len(x)
    return 1.0 - sim_ratio


def normalized_square_euclidean_distance(ranges):
    def actual(x, y, xy_ranges):
        return np.sum(np.square(np.abs(x - y) / xy_ranges))
    return lambda x, y: actual(x, y, ranges)


def mixed_distance(x, y, discrete, continuous, class_name, ddist, cdist):

    xd = [x[att] for att in discrete if att != class_name]
    wd = 0.0
    dd = 0.0
    if len(xd) > 0:
        yd = [y[att] for att in discrete if att != class_name]
        wd = 1.0 * len(discrete) / (len(discrete) + len(continuous))
        dd = ddist(xd, yd)

    xc = np.array([x[att] for att in continuous])
    wc = 0.0
    cd = 0.0
    if len(xd) > 0:
        yc = np.array([y[att] for att in continuous])
        wc = 1.0 * len(continuous) / (len(discrete) + len(continuous))
        cd = cdist(xc, yc)

    return wd * dd + wc * cd


def np_gower_distance(z, x, num_feature_ranges, indices_categorical_features=None):
  l = len(z)
  D_c, D_n = 0, 0

  # categorical features  
  if indices_categorical_features and len(indices_categorical_features) > 0:
    D_c = np.sum( z[indices_categorical_features] != x[indices_categorical_features])

    # also set numerical features
    is_numerical = np.ones(l, bool)
    is_numerical[indices_categorical_features] = False
    z_num = z[is_numerical]
    x_num = x[is_numerical]
  else:
    z_num = z
    x_num = x

  # numerical features contribution
  D_n = np.sum( np.divide( np.abs(z_num - x_num), num_feature_ranges))
  D = 1/l * (D_c + D_n)
  return D


def gower_distance(x, y, discrete, continuous, class_name, feature_intervals):
    xd = [x[att] for att in discrete if att != class_name]
    dd = 0.0
    if len(xd) > 0:
        yd = [y[att] for att in discrete if att != class_name]
        count = 0
        for xi, yi in zip(xd, yd):
            if xi == yi:
                count += 1
        dd = count

    xc = np.array([x[att] for att in continuous])
    feature_names = list(x.keys())
    num_feature_intervals = [fi for i, fi in enumerate(feature_intervals) if feature_names[i] in continuous]
    num_feature_ranges = [fi[1] - fi[0] for fi in num_feature_intervals]
    cd = 0.0
    if len(xc) > 0:
        yc = np.array([y[att] for att in continuous])
        cd = mad_distance(xc, yc, num_feature_ranges)

    return 1/(len(discrete)+len(continuous)) * (dd + cd)


def mad_distance(x, y, mad):
    val = 0.0
    for i in range(len(mad)):
        # print i, 0.0 if mad[i] == 0.0 else 1.0 * np.abs(x[i] - y[i]) / mad[i]
        # print i, np.abs(x[i] - y[i]) / mad[i]
        # val += 0.0 if mad[i] != 0 else 1.0 * np.abs(x[i] - y[i]) / mad[i]
        val += 0.0 if mad[i] == 0.0 else 1.0 * np.abs(x[i] - y[i]) / mad[i]
    # print val
    return val
