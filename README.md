# DybLSPB

```python
import tensorflow as wtf
```

We can now run piczak_cv.py normally, for all 10 folds, or we call from the cmd line

python3 piczak_cv.py X

where X is an integer from 0 to 9. It will then perform the X'th fold of the CV only. Whatever the case, it will store (**for every fold individually**) the best trained weights as TF checkpoint and numpy arrays, and the performance over time.

When all 10 folds have been run (this can now happen on several laptops / AWS accounts in parallel!), we can take the weights from the best fold to move on in the project, as Lars said.

Disclaimer: Maybe Sébastien's plan was to do a procedure like this on the dirty_piczak.py and not on the piczak_cv.py. If that is the case, it will be easy to make these changes also to dirty_piczak.py, or merge both .py files into one file that has a dirty/non-dirty option.