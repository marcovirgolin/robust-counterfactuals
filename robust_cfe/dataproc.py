# ADULT
# info: https://archive.ics.uci.edu/ml/datasets/adult
# (gerry fair version: https://github.com/algowatchpenn/GerryFair)

# COMPAS
# info: https://www.kaggle.com/danofer/compass

# BOSTON HOUSING
# info: https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html

# SOUTH GERMAN CREDIT
# info: https://archive.ics.uci.edu/ml/datasets/South+German+Credit+%28UPDATE%29

# GARMENT PRODUCTIVITY
# info: https://archive.ics.uci.edu/ml/datasets/Productivity+Prediction+of+Garment+Employees

import numpy as np
import pandas as pd




# helper functions
'''
def get_one_perc(intervals, feat_name):
    range = intervals[feat_name][1] - intervals[feat_name][0]
    return range / 100.0

def get_iqr(df, feat_name):
    q1 = df[feat_name].quantile(0.25)                 
    q3 = df[feat_name].quantile(0.75)
    iqr = q3 - q1
    return iqr

def check_abp(feature_names, categorical_feature_names, a,b,p):
    for feat in feature_names:
        assert(p[feat] in [None,'<=','=','>='])
        if feat in categorical_feature_names:
            assert(a[feat] is None or (len(a[feat])>1))
            assert(b[feat] is None or (len(b[feat])>1))
        else:
            assert(a[feat] is None or (a[feat][0] <= 0 and a[feat][1] >= 0))
            assert(b[feat] is None or (b[feat][0] <= 0 and b[feat][1] >= 0))
'''

def check_X_respects_intervals(X, feature_intervals, indices_categorical_features):
    for x in X:
        for i, x_i in enumerate(x):
            if i in indices_categorical_features:
                assert(x_i in feature_intervals[i])
            else:
                assert(x_i >= feature_intervals[i][0] and x_i <= feature_intervals[i][1])



# gets a dataset in my format and re-arranges it for LORE
def make_lore_compliant_dataset(dataset):

    # imports needed only if using this
    from lore.util import recognize_features_type, set_discrete_continuous, label_encode

    df = dataset['df']

    # LORE wants the label to be the first column instead of the last one
    columns = list(df.columns)
    columns = columns[-1:] + columns[:-1]
    df = df[columns]

    possible_outcomes = list(df['LABEL'].unique())
    type_features, features_type = recognize_features_type(df, 'LABEL')
    discrete, continuous = set_discrete_continuous(dataset['feature_names'], type_features, 'LABEL', dataset['categorical_feature_names'], continuous=None)
    idx_features = {i: col for i, col in enumerate(dataset['feature_names'])}

    _, label_encoder = label_encode(df, discrete)

    # check that order of columns matches order of features
    assert(list(df.columns) == ['LABEL'] + dataset['feature_names'])

    lore_dataset = dataset
    # add additional info
    lore_dataset['columns'] = columns
    lore_dataset['class_name'] = 'LABEL'
    lore_dataset['possible_outcomes'] = possible_outcomes
    lore_dataset['type_features'] = type_features
    lore_dataset['features_type'] = features_type
    lore_dataset['discrete'] = discrete
    lore_dataset['continuous'] = continuous
    lore_dataset['idx_features'] = idx_features
    lore_dataset['label_encoder'] = label_encoder


    return lore_dataset


def gimme_boston(datasets_folder="./datasets",return_lore_version=False):
    # 1) Load data set and frame as pandas data frame
    Xy = np.genfromtxt(datasets_folder+"/boston_housing.csv", delimiter=',')
    X = Xy[:,:-1]
    y = Xy[:,-1]
    # make classification like T. Laugel
    y = np.array([1 if x > 26.0 else 0 for x in y]).astype(int)

    feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
    categorical_feature_names = ['CHAS']
    indices_categorical_features = [i for i, f in enumerate(feature_names) if f in categorical_feature_names]

    boston_dict = dict()
    for i, feat in enumerate(feature_names):
        boston_dict[feat] = X[:,i]
    boston_dict['LABEL'] = y

    df = pd.DataFrame(boston_dict)
    
    # 2) Set meaningful min and max intervals for the features
    intervals = dict()
    intervals['CRIM'] = (0.0, 100.0)
    intervals['ZN'] = (0.0, 100.0)
    intervals['INDUS'] = (0.0, 30.0)
    intervals['CHAS'] = (0, 1) # this is category
    intervals['NOX'] = (0.3, 1.0)
    intervals['RM'] = (2,10)
    intervals['AGE'] = (2.0, 100.0)
    intervals['DIS'] = (0.5, 15.0)
    intervals['RAD'] = (1, 24) # index, can be treated as a numerical variable
    intervals['TAX'] = (150.0, 800.0)
    intervals['PTRATIO'] = (10.0, 25.0)
    intervals['B'] = (0.0, 400.0) # the description of this features seems to be incorrect 
    intervals['LSTAT'] = (1.0, 40.0)

    feature_intervals = np.array([np.array(intervals[feat]) for feat in feature_names], dtype=object)

    # 3) Set perturbations and plausibility constraints of action
    perturb, plausib = dict(), dict()

    # CRIM - per capita crime rate by town
    # the municipality can invest in, e.g., police, to reduce criminality; but it might also raise (probably less so if it is a feature we act upon)
    perturb['CRIM'] = {'type':'relative', 'increase':0.05, 'decrease':0.01}
    plausib['CRIM'] = None # the municipality can invest to reduce it, or cut funds to police so it might increase
    
    # ZN - proportion of residential land zoned for lots over 25,000 sq.ft. 
    # perturbations can probably lead to *more* residential land zones appering than disappearing
    perturb['ZN'] = {'type':'relative', 'increase':0.05, 'decrease':0.01}
    plausib['ZN'] = None # the municipality can act to change this situation (e.g., via tax deducations/increases for housing)

    # INDUS - proportion of non-retail business acres per town
    # Similar to ZN, but probably more likely to decrease
    perturb['INDUS'] = {'type':'relative', 'increase':0.01, 'decrease':0.02}
    plausib['INDUS'] = None

    # CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
    # no perturbations
    perturb['CHAS'] = None
    # not mutable
    plausib['CHAS'] = '='

    # NOX - nitric oxides concentration (parts per 10 million)
    # perturbations more likely to increase this
    perturb['NOX'] = {'type':'relative', 'increase':0.05, 'decrease':0.01}
    plausib['NOX'] = None # investments can be done / abandoned to increase it or decrease it

    # RM - average number of rooms per dwelling 
    # might vary because of how inhabitants behave because of several reasons
    perturb['RM'] = {'type':'absolute', 'increase':1, 'decrease':1}
    # assume we can put incentives to change this
    plausib['RM'] = None

    # AGE - proportion of owner-occupied units built prior to 1940
    # assume it can only decrease. If left uncheck tends to decrease because new ones are built and existing fall apart
    perturb['AGE'] = {'type':'relative', 'increase':0, 'decrease':0.05}  # relative
    plausib['AGE'] = '<='

    # DIS - weighted distances to five Boston employment centres
    # assume it is not mutable
    perturb['DIS'] = None
    plausib['DIS'] = '='

    # RAD (this index will likely improve)
    perturb['RAD'] = {'type':'absolute', 'increase':3, 'decrease':0}    #absolute
    plausib['RAD'] = None

    # TAX - full-value property-tax rate per $10,000
    # we can act to decrease or increase taxes. 
    perturb['TAX'] = {'type':'relative', 'increase':0.05, 'decrease':0.05}  # relative
    # perturb['TAX'] = {'type':'absolute', 'increase':1, 'decrease':1}    #absolute
    plausib['TAX'] = None

    # PTRATIO - pupil-teacher ratio by town
    # more likely to increase than decrease (pop growth)
    perturb['PTRATIO'] = {'type':'relative', 'increase':0.05, 'decrease':0.03}  # relative
    # we can invest to change this quantity
    plausib['PTRATIO'] = None

    # B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
    # This is a controversial feature. Let's say we do not act on this; but it can change by itself
    perturb['B'] = {'type':'relative', 'increase':0.05, 'decrease':0.05}  # relative
    plausib['B'] = '='                                                                              

    # LSTAT - % lower status of the population
    # Not sure what is meant by this, is "lower status" meaning poorer/less educated? 
    # this is already a percentange so we treat it as absolute
    perturb['LSTAT'] = {'type':'absolute', 'increase':5, 'decrease':5}    #absolute
    plausib['LSTAT'] = None # investments can be made / stopped to improve / make worse the situtation


    # check that perturb and plausib have been filled
    for feat in feature_names:
        assert(feat in perturb)
        assert(feat in plausib)

    # 4) set label, prep numpy arrays
    X = df[feature_names].to_numpy().astype(float)
    y = df['LABEL'].to_numpy().astype('int')

    # check that all is in order
    assert(list(df.columns) == feature_names + ['LABEL'])
    check_X_respects_intervals(X, feature_intervals, indices_categorical_features)

    dataset = {
        'name': 'boston',
        'best_class' : 0,
        'feature_names': feature_names,
        'categorical_feature_names': categorical_feature_names,
        'indices_categorical_features': indices_categorical_features,
        'df': df,
        'X': X,
        'y': y,
        'feature_intervals': feature_intervals,
        'perturbations' : [perturb[key] for key in feature_names],
        'plausibility_constraints' : [plausib[key] for key in feature_names]
    }

    if return_lore_version:
        dataset = make_lore_compliant_dataset(dataset)

    return dataset

def gimme_compas(datasets_folder="./datasets",return_lore_version=False):

    # 1) Load data and do some preprocessing
    df = pd.read_csv(datasets_folder+'/compas-scores-two-years.csv')
    # I follow R. Guidotti's LORE here:
    # https://github.com/riccotti/LORE/blob/710ffb42bf764bae90e9295e14349f0250fc2628/prepare_dataset.py#L100
    # except for the fact that I also exclude age_cat, since we have age already
    columns = ['age', 'sex', 'race',  'priors_count', 'days_b_screening_arrest', 'c_jail_in', 'c_jail_out',
               'c_charge_degree', 'is_recid', 'is_violent_recid', 'two_year_recid', 'decile_score', 'score_text']
    df = df[columns]

    # like Guidotti
    del df['score_text']

    df['days_b_screening_arrest'] = np.abs(df['days_b_screening_arrest'])
    df['c_jail_out'] = pd.to_datetime(df['c_jail_out'])
    df['c_jail_in'] = pd.to_datetime(df['c_jail_in'])
    df['length_of_stay'] = (df['c_jail_out'] - df['c_jail_in']).dt.days
    df['length_of_stay'] = np.abs(df['length_of_stay'])
    # different from Guidotti, we drop NAs instead of filling them in
    df.dropna(axis=0, inplace=True)
    # like Guidotti, remove jail in and out
    del df['c_jail_in']
    del df['c_jail_out']
    df['length_of_stay'] = df['length_of_stay'].astype(int)
    
    df['days_b_screening_arrest'] = df['days_b_screening_arrest'].astype(int)

    # Guidotti fills missing values, we remove these rows
    #df['length_of_stay'].fillna(df['length_of_stay'].value_counts().index[0], inplace=True)
    #df['days_b_screening_arrest'].fillna(df['days_b_screening_arrest'].value_counts().index[0], inplace=True)
    
    def get_class(x):
        if x < 7:
            return 0
        else:
            return 1

    df['LABEL'] = df['decile_score'].apply(get_class)
    del df['decile_score']

    # keep top 2000 rows only
    df = df[:2000]
    
    feature_names = list(df.columns)
    feature_names.remove('LABEL')

    categorical_feature_names = ['sex', 'race', 'c_charge_degree', 'is_recid', 'is_violent_recid', 'two_year_recid']
    indices_categorical_features = [i for i, f in enumerate(feature_names) if f in categorical_feature_names]
    
    # convert categorical features to codes
    for feat in categorical_feature_names:
        df[feat] = pd.Categorical(df[feat])
        df[feat] = df[feat].cat.codes


    # 2) set up intervals
    intervals = dict()
    # for categorical features, put all possible categories
    for feat in categorical_feature_names:
        intervals[feat] = sorted(df[feat].unique().tolist())
    # next define numerical features
    intervals['age'] = (18, 100)
    intervals['priors_count'] = (0, 100) # the max of the data set is 38 but I guess there is no real limit here
    intervals['days_b_screening_arrest'] = (0, 1200) # the max of the data set is 1057, I cap it at 1200
    intervals['length_of_stay'] = (0, 1200) # the max of the data set is 799, I cap this also at 1200

    # check we inserted all features
    for feat in feature_names:
        assert(feat in intervals)

    feature_intervals = np.array([np.array(intervals[feat]) for feat in feature_names], dtype=object)

    # 3) Set perturbations and plausibility constraints of action
    perturb, plausib = dict(), dict()

    # let us initialize them to None as many features cannot change
    for feat in feature_names:
        perturb[feat] = None
        plausib[feat] = None
        

 
    # Let's assume that we can actively decide to postpone the assessment by the black-box by up to 2 years, this means that age can increase
    perturb['age'] = {'type':'absolute', 'increase':2, 'decrease':0}    #absolute
    plausib['age'] = '>='

    # similarly, let's assume that we can actively decide to stay longer before assessment, by 2*365 days = 730
    perturb['length_of_stay'] = {'type':'absolute', 'increase':730, 'decrease':0}    #absolute
    plausib['length_of_stay'] = '>='
    
    # let's assume that we can be assessed up to two years later because of terrible delays in the system
    perturb['days_before_screening_arrest'] = {'type':'absolute', 'increase':730, 'decrease':0}    #absolute
    plausib['days_before_screening_arrest'] = None

    
    # Priors that were not counted for are discovered at the time of the assessment. 
    # Since the dataset's max is 38, let us say that up to 3 more can be found
    # Let us also say that we might find that some info was incorrect, so priors can lower
    perturb['priors_count'] = {'type':'absolute', 'increase':3, 'decrease': 2}    #absolute
    plausib['priors_count'] = None

    
    # Recidism can become true or false if new priors are discovered, dismissed
    perturb['is_recid'] = {'type':'absolute', 'categories': df['is_recid'].unique() } 
    plausib['is_recid'] = None

    # similar reasoning as above
    perturb['two_year_recid'] = {'type':'absolute', 'categories': df['two_year_recid'].unique() } 
    plausib['two_year_recid'] = None

    # the same could be said for violent_recid but this is unlikely, i.e., unlikely that a prior that was not accounted for was actually a violent one
    perturb['is_violent_recid'] = None
    plausib['is_violent_recid'] = '='

    # sex cannot change
    perturb['sex'] = None
    plausib['sex'] = '='

    # check that perturb and plausib have been filled
    for feat in feature_names:
        assert(feat in perturb)
        assert(feat in plausib)

    # 4) Create numpy array and wrap it up
    X = df[feature_names].to_numpy()
    y = df['LABEL'].to_numpy().astype(int)

    # check that all is in order
    assert(list(df.columns) == feature_names + ['LABEL'])
    check_X_respects_intervals(X, feature_intervals, indices_categorical_features)
    

    dataset = {
        'name': 'compas',
        'best_class' : 0,
        'feature_names': feature_names,
        'categorical_feature_names': categorical_feature_names,
        'indices_categorical_features': indices_categorical_features,
        'df': df,
        'X': X,
        'y': y,
        'feature_intervals': feature_intervals,
        'perturbations' : [perturb[key] for key in feature_names],
        'plausibility_constraints' : [plausib[key] for key in feature_names]
    }

    if return_lore_version:
        dataset = make_lore_compliant_dataset(dataset)


    return dataset

def gimme_adult(datasets_folder="./datasets", return_lore_version=False):
    # 1) Load data and do some preprocessing
    df = pd.read_csv(datasets_folder+'/adult.csv', delimiter=',', skipinitialspace=True)
    # I follow a pre-processing similar to R. Guidotti's LORE:
    # https://github.com/riccotti/LORE/blob/710ffb42bf764bae90e9295e14349f0250fc2628/prepare_dataset.py
    
    # remove so-called useless columns as done by R. Guidotti 
    del df['fnlwgt']
    # Guidotti removes education-num and keeps education, however 
    # education is an index rather than a class, so here we keep the numerical feature
    # (e.g. assuming you can improve your education-num by following specialization courses)
    #del df['education-num']
    del df['education']

    # remove rows with missing Values
    df.replace('?', np.nan, inplace=True)
    df.dropna(axis=0, inplace=True)

    # preprocess categorical features
    categorical_feature_names = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

    # convert categorical features to codes
    for feat in categorical_feature_names:
        df[feat] = pd.Categorical(df[feat])
        df[feat] = df[feat].cat.codes

    # change y's label
    df['LABEL'] = df['income']
    del df['income']

    # all feature names
    feature_names = list(df.columns)
    feature_names.remove('LABEL')
    indices_categorical_features = [i for i, f in enumerate(feature_names) if f in categorical_feature_names]

    # 2) set up intervals
    intervals = dict()
    # let's intilize categorical features
    for feat in categorical_feature_names:
        intervals[feat] = sorted(df[feat].unique().tolist())

    # if you want to print intervals from df:
    #for feat in [x for x in feature_names if x not in categorical_feature_names]:
    #    print(feat, df[feat].min(), df[feat].max())
        
    intervals['age'] = (17, 90) # like min and max from dataset
    intervals['education-num'] = (2, 16) # like min and max from dataset
    intervals['capital-gain'] = (0, 99999) # like min and max from dataset
    intervals['capital-loss'] = (0, 4500) # the max of the "full" dataset is 4356, making it a nice round number
    intervals['hours-per-week'] = (1, 99) # like min and max from "full" dataset

    feature_intervals = np.array([np.array(intervals[feat]) for feat in feature_names], dtype=object)
    
    # 3) Set perturbations and plausibility constraints of action
    perturb, plausib = dict(), dict()


    # let us initialize them in bunch
    for feat in feature_names:
        if 'workclass'==feat or 'marital-status'==feat or 'occupation'==feat or 'relationship'==feat:
            # these can vary, let us assume it is not happening when we decide to act on them
            perturb[feat] = {'type':'absolute', 'categories': df[feat].unique() } 
            plausib[feat] = None
        elif 'race'==feat or 'native-country'==feat:
            # you cannot change your race nor native country
            perturb[feat] = None
            plausib[feat] = '=' # must stay the same
        elif 'sex'==feat:
            # no perturbations possible
            perturb[feat] = None
            plausib[feat] = '=' # must stay the same
        elif feat == 'age':
            # let us say that, if we decide to wait and get older, we can have a delay of half a year happening
            # but if this is not something we actively focus on, delays can cause us to age, say up to 2 years
            perturb[feat] = {'type':'absolute', 'increase':2, 'decrease':0}    #absolute
            # age can only increase
            plausib[feat] = '>='
        elif feat == 'capital-gain':
            # it can happen that we gain less/more than we want due to some circumstances
            # probably more likely to loose gain than to... gain gain
            perturb[feat] = {'type':'relative', 'increase':0.05, 'decrease':0.1}  
            plausib[feat] = None
        elif feat == 'capital-loss':
            # opposite of capital gain
            perturb[feat] = {'type':'relative', 'increase':0.1, 'decrease':0.05}  
            plausib[feat] = None
        elif feat == 'hours-per-week':
            # e.g. need to work less / more for some weeks due to some circumstances
            perturb[feat] = {'type':'absolute', 'increase':5, 'decrease':5}   
            plausib[feat] = None
        elif feat == 'education-num':
            # this does not change by itself, and you can only improve your education level
            perturb[feat] = None 
            plausib[feat] = '>='
        else:
            raise ValueError("Not handled feat", feat)

    # check that perturb and plausib have been filled
    for feat in feature_names:
        assert(feat in perturb)
        assert(feat in plausib)

    # 4) set label, prep numpy arrays
    X = df[feature_names].to_numpy()
    y = df['LABEL'].to_numpy().astype(int)

    # check that all is in order
    assert(list(df.columns) == feature_names + ['LABEL'])
    check_X_respects_intervals(X, feature_intervals, indices_categorical_features)

    dataset = {
        'name': 'adult',
        'best_class' : 1,
        'feature_names': feature_names,
        'categorical_feature_names': categorical_feature_names,
        'indices_categorical_features': indices_categorical_features,
        'df': df,
        'X': X,
        'y': y,
        'feature_intervals': feature_intervals,
        'perturbations' : [perturb[key] for key in feature_names],
        'plausibility_constraints' : [plausib[key] for key in feature_names]
    }

    if return_lore_version:
        dataset = make_lore_compliant_dataset(dataset)

    return dataset

def gimme_credit(datasets_folder="./datasets",return_lore_version=False):
    # 1) Load data set and frame as pandas data frame
    df = pd.read_csv(datasets_folder+'/south_german_credit.csv')

    # Note: some features are categorical indices (larger number = better) and we consider them to be numerical
    # (assuming small changes are possible, of course)
    # these are: 
    '''
    present_emp_since,
    savings,
    account_check_status,
    credit_history,
    housing,
    job,
    property
    '''
    
    # preprocess categorical features
    categorical_feature_names = ['purpose', 'personal_status_sex',
        'other_debtors', 'other_installment_plans', 'telephone', 'foreign_worker']

    for feat in categorical_feature_names:
        df[feat] = pd.Categorical(df[feat])
        df[feat] = df[feat].cat.codes
    
    # change y's label
    df['LABEL'] = df['credit_risk']
    del df['credit_risk']
    # note: in south german credit, as opposed to older version of german credit,
    # credit_risk = 0 means bad, credit_risk = 1 means good

    # set feature names
    feature_names = list(df.columns)
    feature_names.remove('LABEL')

    # keep track of which features are categorical
    indices_categorical_features = [i for i, f in enumerate(feature_names) if f in categorical_feature_names]

    # 2) set up intervals
    intervals = dict()

    # intervals for categorical features to their possibilities
    for feat in categorical_feature_names:
        intervals[feat] = sorted(df[feat].unique().tolist())

    # set numerical feature intervals
    # let's just initialize them to max and min in the data, then refine some
    for feat in feature_names:
        if feat not in categorical_feature_names:
            intervals[feat] = (df[feat].min(), df[feat].max())

    intervals['age'] = (18, 75) # min of data set is 19, max is 75
    intervals['duration_in_month'] = (3, 75) # data set min and max are 4, 72, let's allow a bit more wiggle room
    intervals['credit_amount'] = (250, 20000) # data set min and max are 250, 18,424; the latter seems arbitrary so let's cap at 20,000 instead
    

    # check we set everything
    for feat in feature_names:
        assert (feat in intervals)

    feature_intervals = np.array([np.array(intervals[feat]) for feat in feature_names], dtype=object)


    # 3) Set perturbations and plausibility constraints of action
    perturb, plausib = dict(), dict()

    # let's start with numerical features

    # can get older, say up to half-a-year of delay in the assessment
    # cannot get younger
    perturb['age'] = {'type':'absolute', 'increase':0.5, 'decrease':0}   
    plausib['age'] = '>='

    # the income might vary and thus this quantity
    perturb['installment_as_income_perc'] = {'type':'relative', 'increase':0.10, 'decrease':0.10}  
    plausib['installment_as_income_perc'] = None
    
    # how long one has been at the current residence
    # cannot vary without our control (sure one could be evicted but let's assume that's extremely rare)
    perturb['present_res_since'] = None 
    # this can grow or drop (e.g., to 0) if one relocates; for simplicity let's just assume that one does not change residence (hence >=)
    plausib['present_res_since'] = '>=' 

    # duration of the credit
    # assume perturbations can change it a bit relative to what was asked for
    # (but less likely to decrease than increase)
    perturb['duration_in_month'] = {'type':'relative', 'increase':0.25, 'decrease':0.05} 
    plausib['duration_in_month'] = None 
    
    # assume purpose does not change
    perturb['purpose'] = None
    # also: not plausible to change purpose
    # (pretty useless to know that if you want a house instead of a yacht, then they give you the credit for that)
    plausib['purpose'] = '='

    # housing type: assume no perturbations similar to present_res_since
    perturb['housing'] = None
    # but let's say it is plausible to change house
    plausib['housing'] = None

    # series of features which might vary
    for feat in ['account_check_status','credit_this_bank']:
        perturb[feat] = {'type':'absolute', 'increase':1, 'decrease':1}
        plausib[feat] = None
    
    # plausible to go both ways (we can ask less or more credit)
    # let's say that events may happen that require us to ask a bit more or a bit less
    perturb['credit_amount'] = {'type':'relative', 'increase':0.1, 'decrease':0.1} 
    plausib['credit_amount'] = None

    perturb['savings'] = {'type':'relative', 'increase':0.1, 'decrease':0.1} 
    plausib['savings'] = None

    # similar reasoning to present_res_since
    perturb['present_emp_since'] = None
    plausib['present_emp_since'] = '>='

    # some features we assume cannot vary due to external events but are plausible to intentionally change
    for feat in ['other_installment_plans', 'credits_this_bank', 'job']:
        perturb[feat], plausib[feat] = None, None

    # some features that we assume cannot vary nor change
    for feat in ['credit_history','personal_status_sex','other_debtors','property', 
        'telephone', 'foreign_worker', 'people_under_maintenance']:
        perturb[feat], plausib[feat] = None, '='

    # check we filled them all
    for feat in feature_names:
        assert(feat in perturb)
        assert(feat in plausib)


    # 4) set label, prep numpy arrays
    X = df[feature_names].to_numpy()
    y = df['LABEL'].to_numpy().astype(int)

    # check that all is in order
    assert(list(df.columns) == feature_names + ['LABEL'])
    check_X_respects_intervals(X, feature_intervals, indices_categorical_features)
    

    dataset = {
        'name': 'credit',
        'best_class' : 1,
        'feature_names': feature_names,
        'categorical_feature_names': categorical_feature_names,
        'indices_categorical_features': indices_categorical_features,
        'df': df,
        'X': X,
        'y': y,
        'feature_intervals': feature_intervals,
        'perturbations' : [perturb[key] for key in feature_names],
        'plausibility_constraints' : [plausib[key] for key in feature_names]
    }

    if return_lore_version:
        dataset = make_lore_compliant_dataset(dataset)

    return dataset

def gimme_garments(datasets_folder="./datasets", return_lore_version=False):
    # 1) Load data and do some preprocessing
    df = pd.read_csv(datasets_folder+"/garments_worker_productivity.csv", delimiter=',')
    # remove date and wip (a lot of missing values in the latter)
    df.drop(['date','wip'], axis=1, inplace=True)
    # remove duplicate rows
    mask_unique = [not d for d in df.drop('actual_productivity', axis=1).duplicated()]
    df = df[mask_unique]

    # create (quasi-)balanced ternary label
    df['LABEL'] = 0
    df.loc[df.actual_productivity > .7, 'LABEL'] = 1
    df.loc[df.actual_productivity > .81, 'LABEL'] = 2
    del df['actual_productivity']
    

    feature_names = df.columns.to_list()
    feature_names.remove('LABEL')
    categorical_feature_names = ['quarter', 'department', 'day', 'team', 'no_of_style_change']
    indices_categorical_features = [i for i, f in enumerate(feature_names) if f in categorical_feature_names]

    # convert categorical features to codes
    for feat in categorical_feature_names:
        df[feat] = pd.Categorical(df[feat])
        df[feat] = df[feat].cat.codes

    # 2) Set meaningful min and max intervals for the features
    # (for categoricals we store all categories)
    intervals = dict()

    for feat in categorical_feature_names:
        intervals[feat] = sorted(df[feat].unique().tolist())


    intervals['smv'] = (2.5, 60) # similar to min and max, rounded them a bit
    intervals['over_time'] = (0, 25920) # the max of the data set is 25920 min = 18 days
    intervals['incentive'] = (0, 3600) # just like min and max of data set
    intervals['idle_time'] = (0, 300) # like min and max of the data set
    intervals['idle_men'] = (0, 50) # min and max are 0 45, we round the latter up
    intervals['no_of_workers'] = (1, 100) # min and max are 2 89, we imagine it can be 1 to 100
    intervals['targeted_productivity'] = (0.05, 0.8) # min and max are 0.07 and 0.8
    
    # check we set them all
    for feat in feature_names:
        assert(feat in intervals.keys())

    feature_intervals = np.array([np.array(intervals[feat]) for feat in feature_names], dtype=object)
    
    # 3) Set perturbations and plausibility constraints of action
    # we assume we are the company manager, who acts to improve their workers' productivity
    perturb, plausib = dict(), dict()


    # Team: Associated team number with the instance
    # Cannot be perturbed, but it is actionable if we assume that teams can be re-assigned
    perturb['team'] = None
    plausib['team'] = None

    # day : Day of the Week
    # let's assume that delays can change the scheduled day for production to change
    perturb['day'] = {'type':'absolute', 'categories': df['day'].unique() }
    # we can decide to change the day to do a certain task as we like
    plausib['day'] = None
    
    # quarter: A portion of the month. A month was divided into four quarters
    # similar reasoning to day
    perturb['quarter'] = {'type':'absolute', 'categories': df['quarter'].unique() } 
    # we can decide to change the quarter as we like
    plausib['quarter'] = None
      
    # department: Associated department with the instance
    # we assume they cannot vary beyond our control
    perturb['department'] = None
    # we can vary it as we like
    plausib['department'] = None
       
    # no_of_workers: Number of workers in each team
    # workers might quit/become ill for example. they cannot be hired without we willingly want to.
    perturb['no_of_workers'] = {'type':'relative', 'increase':0, 'decrease':0.1} 
    # assume we can let off or hire personell
    plausib['no_of_workers'] = None

    # no_of_style_change : Number of changes in the style of a particular product
    # assume this cannot be perturbed
    perturb['no_of_style_change'] = None 
    # we can act to modify how much style changes there should be
    plausib['no_of_style_change'] = None  
    
    # targeted_productivity : Targeted productivity set by the Authority for each team for each day.
    # it's a fixed goal, cannot be perturbed but can be changed willingly
    perturb['targeted_productivity'] = None
    plausib['targeted_productivity'] = None
    
    # smv : Standard Minute Value, it is the allocated time for a task
    # Let's assume the "real" smv is subject to perturbations
    perturb['smv'] = {'type':'relative', 'increase':0.1, 'decrease':0.1} 
    # we can change how much time we think should be allocated for a task
    plausib['smv'] = None
    
    # over_time : Represents the amount of overtime by each team in minutes
    # Let's assume that this can be subject to perturbations
    # max is 25920 min = 18 days. Let's say perturbations can cause up to +-3 days
    perturb['over_time'] = {'type':'absolute', 'increase':3*24*60, 'decrease':3*24*60} 
    # We can demand more or less overtime
    plausib['over_time'] = None
    
    # incentive : Represents the amount of financial incentive (in BDT) that enables or motivates a particular course of action.
    # Assume it is not subject to perturbations
    perturb['incentive'] = None
    # the company can change it 
    plausib['incentive'] = None

    # idle_time : The amount of time when the production was interrupted due to several reasons
    # Probably more likely that idle time increses rather than decreases due to perturbations
    perturb['idle_time'] = {'type':'relative', 'increase':0.1, 'decrease':0.05}  
    # we assume the company can act to change this
    plausib['idle_time'] = None
    
    # idle_men : The number of workers who were idle due to production interruption
    # Same reasoning as for idle_time
    perturb['idle_men'] = {'type':'relative', 'increase':0.1, 'decrease':0.05}  
    # can take actions to change this
    plausib['idle_men'] = None
    

    # check that we filled them all
    for feat in feature_names:
        assert(feat in perturb)
        assert(feat in plausib)    

    # 4) set label, prep numpy arrays
    X = df[feature_names].to_numpy().astype(float)
    y = df['LABEL'].to_numpy().astype('int')

    # check that all is in order
    assert(list(df.columns) == feature_names + ['LABEL'])

    dataset = {
        'name': 'garments',
        'best_class' : 2,
        'feature_names': feature_names,
        'categorical_feature_names': categorical_feature_names,
        'indices_categorical_features': indices_categorical_features,
        'df': df,
        'X': X,
        'y': y,
        'feature_intervals': feature_intervals,
        'perturbations' : [perturb[key] for key in feature_names],
        'plausibility_constraints' : [plausib[key] for key in feature_names]
    }

    if return_lore_version:
        dataset = make_lore_compliant_dataset(dataset)

    return dataset

def gimme_student(datasets_folder="./datasets", return_lore_version=False):

    # 1) Load data and do some preprocessing
    df = pd.read_csv(datasets_folder+"/student_mat.csv", delimiter=',')
    # remove unused labels G1 & G2
    df.drop(['G1','G2'], axis=1, inplace=True)

    # create ternary label (balanced)
    df['G3'] = df['G3'].astype(int)
    df['LABEL'] = 0
    df.loc[df['G3'] > 9, 'LABEL'] = 1
    df.loc[df['G3'] > 12, 'LABEL'] = 2
    del df['G3']

    feature_names = df.columns.to_list()
    feature_names.remove('LABEL')
    categorical_feature_names = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 
        'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'failures']
    indices_categorical_features = [i for i, f in enumerate(feature_names) if f in categorical_feature_names]

    # convert categorical features to codes
    for feat in categorical_feature_names:
        df[feat] = pd.Categorical(df[feat])
        df[feat] = df[feat].cat.codes
    
    # 2) Set meaningful min and max intervals for the features
    # (for categoricals we store all categories)
    intervals = dict()

    for feat in categorical_feature_names:
        intervals[feat] = sorted(df[feat].unique().tolist())
    
    intervals['age'] = (15, 24) # similar to min and max, allowed for a slightly larger max
    intervals['Medu'] = (0, 4) # min and max 
    intervals['Fedu'] = (0, 4) # min and max
    intervals['traveltime'] = (1, 4) # min and max
    intervals['studytime'] = (1, 4) # min and max
    intervals['famrel'] = (1, 5) # min and max
    intervals['freetime'] = (1, 5) # min and max
    intervals['goout'] = (1, 5) # min and max
    intervals['Dalc'] = (1, 5) # min and max
    intervals['Walc'] = (1, 5) # min and max
    intervals['health'] = (1, 5) # min and max
    intervals['absences'] = (0, 100) # similar to min (1) and max (93)
    
    # check we set them all
    for feat in feature_names:
        assert(feat in intervals.keys())

    feature_intervals = np.array([np.array(intervals[feat]) for feat in feature_names], dtype=object)
    
    # 3) Set a's, b's, and plausibility constraints
    # we assume we are the company manager, who acts to improve their workers' productivity
    perturb, plausib = dict(), dict()
    
    # 1 school 
    # the type of school cannot be changed by a perturbation
    perturb['school'] = None
    # we assume the student can change school
    plausib['school'] = None
    
    
    # 2 sex - student's sex
    # of course cannot change by itself
    perturb['sex'] = None
    # a person can change gender but we doubt that's a plausible recommendation
    plausib['sex'] = '='
    
    # 3 age - student's age
    # external circumstances can cause delays (e.g., the student is ill and skips one year)
    perturb['age'] = {'type':'absolute', 'increase':1, 'decrease':0} 
    # age can only increase
    plausib['age'] = '>='
       
    # 4 address - student's home address type (binary: "U" - urban or "R" - rural)
    # the family might decide to relocate irrespective of the situation of the student
    perturb['address'] = {'type':'absolute', 'categories': df['address'].unique() }
    # the family can in fact decide to act i.e. move, to e.g. have the student be closer to the school 
    plausib['address'] = None
       
    # 5 famsize - family size (binary: "LE3" - less or equal to 3 or "GT3" - greater than 3)
    # Events might change the size of the family itself (e.g., a baby is born, a grandparent passes)
    perturb['famsize'] = {'type':'absolute', 'categories': df['famsize'].unique() } 
    # one does not take action on increasing or decreasing the family size to improve a student's grade
    plausib['famsize'] = "="    
    
    # 6 Pstatus - parent's cohabitation status (binary: "T" - living together or "A" - apart)
    # might change by itself, for the better or worse...
    perturb['Pstatus'] = {'type':'absolute', 'categories': df['Pstatus'].unique() } 
    # Similarly to famsize, hard to prescribe parents to get back together...
    plausib['Pstatus'] = "="

    # For the following features, we assume they cannot really change by perturbations
    # nor the parents are likely gonna make it to act sufficiently fast for these to change
    # (note: jobs can become "at_home" if parents get fired, but let's positively assume that's very unlikely)
    for feat in ['Medu','Fedu','Mjob','Fjob', 'reason', 'failures']:
        perturb[feat], plausib[feat] = None, '='
        
    # 12 guardian - student's guardian (nominal: "mother", "father" or "other")
    # Might change if parents separate, one passes, etc.
    perturb['guardian'] = {'type':'absolute', 'categories': df['guardian'].unique() }
    # Might willingly want to change that (e.g., recommend the student to live with the (sperated) father if closer to school)
    plausib['guardian'] = None
    
    # 13 traveltime - home to school travel time (numeric: 1 - <15 min., 2 - 15 to 30 min., 3 - 30 min. to 1 hour, or 4 - >1 hour)
    # Let's say that can be reduced or increased because of several reasons
    # Less change much by perturbations if this is something we act upon
    perturb['traveltime'] = {'type':'absolute', 'increase':3, 'decrease':3}    
    # let's assume that we can act to reduce this (e.g., by a different means of transporation)
    plausib['traveltime'] = None
    
    # 14 studytime - weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)
    # Assume perturbations can only make this worse
    perturb['studytime'] = {'type':'absolute', 'increase':0, 'decrease':3}  
    plausib['studytime'] = None
    
    # 16 schoolsup - extra educational support (binary: yes or no)
    # Assume actionable and also mutable by perturbations
    perturb['schoolsup'] = {'type':'absolute', 'categories': df['schoolsup'].unique() } 
    plausib['schoolsup'] = None
    
    # 17 famsup - family educational support (binary: yes or no)
    # like above probably
    perturb['famsup'] = {'type':'absolute', 'categories': df['famsup'].unique() } 
    plausib['famsup'] = None

    # assume that the following cannot have perturbations but can be changed
    for feat in ['paid','activities','nursery','higher','internet']:
        perturb[feat], plausib[feat] = None, None
    
    # 23 romantic - with a romantic relationship (binary: yes or no)
    # can change by perturbations beyond the student or student's family control
    perturb['romantic'] = {'type':'absolute', 'categories': df['romantic'].unique() } 
    # not so plausible to recommend a student to change his/her relationship status (or is it?)
    plausib['romantic'] = '='
    
    # 24 famrel - quality of family relationships (numeric: from 1 - very bad to 5 - excellent)
    # let us assume that can change and acted upon to improve
    perturb['famrel'] = {'type':'absolute', 'increase':1, 'decrease':1}
    plausib['famrel'] = None
    
    # 25 freetime - free time after school (numeric: from 1 - very low to 5 - very high)
    # similar reasoning to famrel 
    perturb['freetime'] = {'type':'absolute', 'increase':1, 'decrease':1}  
    plausib['freetime'] = None
    
    # 26 goout - going out with friends (numeric: from 1 - very low to 5 - very high)
    # similar to famrel
    perturb['goout'] = {'type':'absolute', 'increase':2, 'decrease':1}  
    plausib['goout'] = None
    
    # 27 Dalc - workday alcohol consumption (numeric: from 1 - very low to 5 - very high)
    perturb['Dalc'] = {'type':'absolute', 'increase':2, 'decrease':1}
    plausib['Dalc'] = None
    
    # 28 Walc - weekend alcohol consumption (numeric: from 1 - very low to 5 - very high)
    perturb['Walc'] = {'type':'absolute', 'increase':2, 'decrease':1}   
    plausib['Walc'] = None
    
    # 29 health - current health status (numeric: from 1 - very bad to 5 - very good)
    perturb['health'] = {'type':'absolute', 'increase':1, 'decrease':3} 
    plausib['health'] = None
    
    # 30 absences - number of school absences (numeric: from 0 to 93)
    perturb['absences'] = {'type':'absolute', 'increase':10, 'decrease':10} 
    plausib['absences'] = None
      
    # check that we filled info for all features
    for feat in feature_names:
        assert(feat in perturb)
        assert(feat in plausib)

    # 4) set label, prep numpy arrays
    X = df[feature_names].to_numpy().astype(float)
    y = df['LABEL'].to_numpy().astype('int')

    # check that all is in order
    assert(list(df.columns) == feature_names + ['LABEL'])

    dataset = {
        'name': 'student',
        'best_class' : 2,
        'feature_names': feature_names,
        'categorical_feature_names': categorical_feature_names,
        'indices_categorical_features': indices_categorical_features,
        'df': df,
        'X': X,
        'y': y,
        'feature_intervals': feature_intervals,
        'perturbations' : [perturb[key] for key in feature_names],
        'plausibility_constraints' : [plausib[key] for key in feature_names]
    }

    if return_lore_version:
        dataset = make_lore_compliant_dataset(dataset)

    return dataset


name_to_dataset = {
    'adult'  : gimme_adult,
    'boston' : gimme_boston,
    'credit' : gimme_credit,
    'compas' : gimme_compas,
    'garments' : gimme_garments,
    'student' : gimme_student,
}


def gimme(dataset_name, datasets_folder="./datasets", return_lore_version=False):
    f = name_to_dataset[dataset_name]
    return f(datasets_folder, return_lore_version)
