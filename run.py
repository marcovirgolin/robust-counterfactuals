import numpy as np
import argparse, os
import time
from joblib import Parallel, delayed, dump, load
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from robust_cfe.dataproc import *
from robust_cfe.wrappers import *
from robust_cfe.blackbox_with_preproc import BlackboxWithPreproc


# set overall seed
OVERALL_SEED = 42
N_FOLDS = 5
np.random.seed(OVERALL_SEED)

def str2bool(v):
  # from https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
  return v.lower() in ("yes", "true", "y", "t", "1")

def parse_options():
  parser = argparse.ArgumentParser("Arguments for the experiments")

  parser.add_argument("--dataset", type=str, help="name of the dataset to consider")
  parser.add_argument("--method", type=str, default="cogs", help="method to use to discover counterfactual examples")
  parser.add_argument("--n_reps", type=int, default=5, help="number of times to repeat the execution of the method")
  parser.add_argument("--n_jobs", type=int, default=2, help="number of jobs for parallel execution")
  parser.add_argument("--check_plausibility", type=str2bool, default=False, help="take into account plausibility constraints")
  parser.add_argument("--optimize_C_robust", type=str2bool, default=False, help="optimize for worst-case C-setbacks")
  parser.add_argument("--optimize_K_robust", type=int, default=0, help="number of samples to optimize for K-robustness (ignored if 0)")
  parser.add_argument("--run_only_fold", type=int, default=None, help="if set, runs only that fold (default is None)")
  parser.add_argument("--n_samples_fold", type=int, default=9999999, help="How many test samples to consider per fold")


  opt = parser.parse_args()
  return opt

# method wrappers for dynamic load up
method_wrappers = {
  'cogs' : CogsWrapper,
  'lore' : LOREWrapper,
  'nelder-mead' : ScipyOptWrapper,
  'growingspheres': GrowingSpheresWrapper,
  'cma' : CMAWrapper,
  'dice-genetic' : DiCEWrapper,
  'dice-random' : DiCEWrapper,
}
# the following must be defined dynamically
kwargs_wrappers = {
  "dice-genetic" : {"method":"genetic"},
  "dice-random" : {"method":"random"}
}


# read in parameters
opt = parse_options()

# load data
dataset = gimme(opt.dataset)
X = dataset['X']
y = dataset['y']

# set plausib constraints to None if opt.check_plausibility is False
if opt.check_plausibility is False:
  dataset['plausibility_constraints'] = [None] * len(dataset['plausibility_constraints'])

result_folder = "results/dataset_"+opt.dataset+"_dclass_"+str(dataset['best_class'])+"_method_"+opt.method
result_folder += "_checkplausib_"+str(opt.check_plausibility)+"_optCrobust_"+str(opt.optimize_C_robust)+"_optKrobust_"+str(opt.optimize_K_robust)
os.makedirs(os.path.join(result_folder), exist_ok=True)

# load data
dataset = gimme(opt.dataset)
X = dataset['X']
y = dataset['y']
if opt.check_plausibility is False:
  dataset['plausibility_constraints'] = [None] * len(dataset['plausibility_constraints'])

# hyper-params black-box
random_forest_param_grid = {
  'blackbox__n_estimators': (50,500),
  'blackbox__min_samples_split' : (2, 8),
  'blackbox__max_features' : ('auto', None)
}
rf = BlackboxWithPreproc(RandomForestClassifier(random_state=OVERALL_SEED), dataset['indices_categorical_features'], preprocs=['onehot'])
gcv = GridSearchCV(rf, param_grid=random_forest_param_grid, refit=True, cv=5, n_jobs=opt.n_jobs)


# start, let's do k fold
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=OVERALL_SEED)
skf.get_n_splits(X, y)


def try_fetching_prefitted_blackbox(blackbox_dump):
  if os.path.exists(blackbox_dump):
    print("Loading pre-fitted blackbox from", blackbox_dump)
    blackbox = load(blackbox_dump)
    return blackbox
  return None


for fold_idx, (train_index, test_index) in enumerate(skf.split(X, y)):

  if (opt.run_only_fold is not None) and (fold_idx != opt.run_only_fold):
    continue

  print("Running method",opt.method,"on dataset",opt.dataset,"fold",fold_idx,
    "(check plaus:",opt.check_plausibility, "- opt. C robust:",opt.optimize_C_robust, 
    "- opt. K robust:",opt.optimize_K_robust,")")

  df = pd.DataFrame(columns=['dataset','desired_class','overall_seed','fold_idx','test_sample_idx',
    'rep_idx','x','pred_class_x','true_class_x','z','pred_class_z','check_plausibility','opt_C_robust','opt_K_robust','run_time'])

  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]

  # fit the blackbox (try loading it first if was created before)
  if not os.path.exists("results/blackboxes"):
    os.makedirs("results/blackboxes")
  blackbox_dump = "results/blackboxes/blackbox_prob_"+opt.dataset+"_fold_"+str(fold_idx)+".joblib"
  blackbox = try_fetching_prefitted_blackbox(blackbox_dump)
  if blackbox is None:
    gcv.fit(X_train, y_train)
    blackbox = gcv.best_estimator_
    # store the blackbox 
    dump(blackbox, blackbox_dump)

  p_test = blackbox.predict(X_test)
  test_acc = accuracy_score(y_test, p_test)

  # skip cases f(x)=c^*
  test_sample_indices_to_consider = [i for i in range(len(p_test)) if p_test[i] != dataset['best_class']]

  # find counterfactuals
  def process_test_sample(test_sample_idx):  

    x = X_test[test_sample_idx,:]
    pred_class_x = blackbox.predict([x])[0]
    true_class_x = y_test[test_sample_idx]
    
    MethodWrapper = method_wrappers[opt.method]
    mkwargs = None
    if opt.method in kwargs_wrappers:
      mkwargs = kwargs_wrappers[opt.method]

    method = MethodWrapper(x, dataset, blackbox, dataset['best_class'], 
      check_plausibility=opt.check_plausibility,
      optimize_C_robust=opt.optimize_C_robust,
      optimize_K_robust=opt.optimize_K_robust,
      method_kwargs=mkwargs)

    records = list()

    for rep_idx in range(opt.n_reps):
      start_time = time.time()
      z = method.find_cfe()
      run_time = np.round(time.time() - start_time,3)
      pred_class_z = blackbox.predict([z])[0]
      
      # store info
      record = {
        'dataset': opt.dataset,
        'desired_class' : dataset['best_class'],
        'overall_seed' : OVERALL_SEED,
        'fold_idx' : fold_idx, 
        'blackbox_test_acc' : test_acc,
        'test_sample_idx': test_sample_idx,
        'rep_idx' : rep_idx,
        'x': x.tolist(), 
        'pred_class_x' : pred_class_x,
        'true_class_x' : true_class_x,
        'z' : z.tolist(),
        'pred_class_z' : pred_class_z,
        'check_plausibility' : opt.check_plausibility,
        'opt_C_robust' : opt.optimize_C_robust,
        'opt_K_robust' : opt.optimize_K_robust,
        'run_time' : run_time,
      }

      records.append(record)
    
    return records

  # cap to the number we want to consider
  if opt.n_samples_fold < len(test_sample_indices_to_consider):
    test_sample_indices_to_consider = test_sample_indices_to_consider[:opt.n_samples_fold]

  rrecords = Parallel(n_jobs=opt.n_jobs)(delayed(process_test_sample)(i) for i in test_sample_indices_to_consider)
  records = [record for records in rrecords for record in records]

  for record in records:
    df = df.append(record, ignore_index=True)

  df.to_csv(result_folder+"/result_fold_"+str(fold_idx)+".csv", index=False)
