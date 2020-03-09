# Practice XGBoost Demo

## Test the package

## Test Environment

Sharing an environment
You may want to share your environment with someone else---for example, so they can re-create a test that you have done. To allow them to quickly reproduce your environment, with all of its packages and versions, give them a copy of your environment.yml file.

Exporting the **environment.yml** file:

https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#create-env-file-manually

```bash
conda env export > environment.yml
```

Creating an environment from an environment.yml file
Use the terminal or an Anaconda Prompt for the following steps:

Create the environment from the environment.yml file:

```bash
conda env create -f environment.yml
```

## Reference

(xgboost)[https://xgboost.readthedocs.io/en/latest/tutorials/model.html]

(xgboots)[https://xgboost.readthedocs.io/en/latest/get_started.html]

(case study)[https://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn/]

(graphviz issue)[https://stackoverflow.com/questions/35064304/runtimeerror-make-sure-the-graphviz-executables-are-on-your-systems-path-aft]
Solved: on windows add the following code in the **.py**

```python
import os
os.environ["PATH"] += os.pathsep + 'D:/Program Files (x86)/Graphviz2.38/bin/'
```

## Treelite packages demo
