import os

for f in os.listdir("."):
  if f not in ["rename.py", "old_nelder"]:
    g = f.replace("_method_", "_blackbox_rf_method_")
    os.system(f"mv {f} {g}")