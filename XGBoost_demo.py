# %% Step 1: load packages
import os
import xgboost as xgb
import numpy as np


# %% Step 2: read in data
# dtrain = xgb.DMatrix('demo/data/agaricus.txt.train')
# dtest = xgb.DMatrix('demo/data/agaricus.txt.test')

data = np.random.rand(5, 10)  # 5 entities, each contains 10 features
label = np.random.randint(2, size=5)  # binary target
dtrain = xgb.DMatrix(data=data, label=label)

# %% Step 3: train the model
# specify parameters via map
param = {'max_depth': 2, 'eta': 1, 'objective': 'binary:logistic'}
num_round = 2
bst = xgb.train(param, dtrain, num_round)

# # %% Step 4: save the model
# bst.save_model('0001.model')
# bst
# # %% Step 5: reload the model
# del bst
# bst = xgb.Booster()  # init model
# bst.load_model('0001.model')  # load data
# bst
# %% Step 6: test the model / make prediction
data = np.random.rand(7, 10)
dtest = xgb.DMatrix(data)
ypred = bst.predict(dtest)

# %% Step 7: view the model
# plot importance: false!
# xgb.plot_importance(bst)

# %% plot the output tree

# debug
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

xgb.plot_tree(bst)

# %% converts the target tree to a graphviz instance in IPython
xgb.to_graphviz(bst)


# %%
