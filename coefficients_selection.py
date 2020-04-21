import subprocess
import numpy as np


if __name__ == "__main__":
    lasso_l, lasso_r = (-6, -4)
    bins_lasso = (lasso_r - lasso_l) * 2 + 1
    group_lasso_l, group_lasso_r = (-6, -4)
    bins_group_lasso = (group_lasso_r - group_lasso_l) * 2 + 1
    
    lasso_space = np.logspace(lasso_l, lasso_r, bins_lasso)
    group_lasso_space = np.logspace(group_lasso_l, group_lasso_r, bins_lasso)
    
    group_lasso_space = np.logspace(-5, -3, 5)
    
    for i in range(bins_lasso):
        for j in range(bins_group_lasso):
            lasso = lasso_space[i]
            group_lasso = group_lasso_space[j]
            
            cmd = f'python train.py --lasso {lasso} --grouplasso {group_lasso}'
            try:
                    print(subprocess.check_output([cmd], shell=True))
            except subprocess.CalledProcessError as err:
                print(err)
