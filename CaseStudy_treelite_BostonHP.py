# %% Import Pkgs and Data
import xgboost
import treelite
from sklearn.datasets import load_boston
X, y = load_boston(return_X_y=True)
print(f'dimensions of X = {X.shape}')
print(f'dimensions of y = {y.shape}')


# %% Train a tree ensemble model using XGBoost
dtrain = xgboost.DMatrix(X, label=y)
params = {'max_depth': 3, 'eta': 1, 'silent': 1, 'objective': 'reg:linear',
          'eval_metric': 'rmse'}
bst = xgboost.train(params, dtrain, 20, [(dtrain, 'train')])

# %% Pass XGBoost model into treelite¶
model = treelite.Model.from_xgboost(bst)


# %% Generate shared library
toolchain = 'clang'
model.export_lib(toolchain=toolchain, libpath='./model.dylib', verbose=True)


# %%
