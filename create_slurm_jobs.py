import shutil

methods = ['dice-genetic']
datasets = ['boston','adult','compas','credit','garments']
folds = [0,1,2,3,4,5]
plausibs = [0]
optCs = [0]
optKs = [0]

# create configs
configs = list()
for method in methods:
  for dataset in datasets:
    for plaus in plausibs:
      for optC in optCs:
        for optK in optKs:
          for fold in folds:
            config = {
              'method' : method,
              'dataset' : dataset, 
              'plaus' : str(plaus),
              'optC' : str(optC),
              'optK' : str(optK),
              'fold' : str(fold),
            }
            configs.append(config)

for config in configs:
  # create job file
  job_name = ""
  for k in config:
    job_name += k+"_"+config[k]+"_"
  job_name = job_name[:-1] + ".job"
  shutil.copy('template.job', 'slurm_jobs/'+job_name)

  # modify job file
  f = open('slurm_jobs/'+job_name, 'r')
  ll = f.readlines()
  f.close()

  is_param_area = False
  for i, l in enumerate(ll):

    # remove "\n" for the comparisons
    l = l.replace("\n","")

    if l == "# START PARAMS":
      # start writing params
      is_param_area = True
      continue
    elif l == "# END PARAMS":
      # finished
      break

    # set the params
    for k in config:
      if l.replace("\n","") == k.upper():
        # found the param to set, set it!
        ll[i] = k.upper()+"="+config[k]+"\n"
        break

  # write updated file
  f = open('slurm_jobs/'+job_name, 'w')
  for l in ll:
    f.write(l)
  f.close()
