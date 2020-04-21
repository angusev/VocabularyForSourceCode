import subprocess
import numpy as np


if __name__ == "__main__":
    thresh_l, thresh_r = (-2, 0)
    bins_thresh = (thresh_r - thresh_l) * 2 + 1
    thresh_space = np.logspace(thresh_l, thresh_r, bins_thresh)
    
    lasso = 1e-5
    group_lasso = 1e-4
    
    for i in range(bins_thresh):
        threshold = thresh_space[i]
        cmd = f'python train.py --lasso {lasso} --grouplasso {group_lasso} --threshold {threshold}'
        try:
            print(subprocess.check_output([cmd], shell=True))
        except subprocess.CalledProcessError as err:
            print(err)

