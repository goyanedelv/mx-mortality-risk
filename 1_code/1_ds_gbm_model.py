from typing import Dict, List
import pandas as pd
import numpy as np
import yaml
from collections import Counter
from matplotlib import pyplot
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn import preprocessing
from datetime import datetime, time
from pandas._libs.lib import is_integer
from pandas_profiling import ProfileReport


print('Initializing risk model')

# Read parameters file
with open('2_parameters/parameters.yml', 'r') as g:
    parameters = yaml.load(g, Loader=yaml.FullLoader)

## example:
parameters['test_size']

# Read data
print('Reading data...')
data = pd.read_csv('0_data/3_primary/data_for_ml.csv', engine='python')

# Drop rows with weird ages and weird locations
sex_info = data.groupby(['sexo', 'death']).aggregate({'factor' : 'sum'}).reset_index()
sex_info.to_excel('0_data/8_reporting/sexes.xlsx', index = False)

condicion_edad = (data.edad == -7982) | (data.edad == 999)
condicion_local = (data.region == 99999)
condicion_sexo = (data.sexo == 9)

data = data.loc[ -condicion_edad,]
data = data.loc[ -condicion_local,]
data = data.loc[ -condicion_sexo,]

data.death.sum()

if parameters['data_profiling']:
	print('Profiling data...')
	profile = ProfileReport(data, title="Mexico: Life and Death - Kanguro")
	profile.to_file("0_data/8_reporting/data_profile_report.html")

# create X and y dfs
X = data.drop(['causa_nombre', 'death', 'state_name', 'municipio'], axis = 1)
y = data['death']

# Encode categorical values
lbl = preprocessing.LabelEncoder()
X['ocupacion'] = lbl.fit_transform(X['ocupacion'].astype(str))
X['edo_civil'] = lbl.fit_transform(X['edo_civil'].astype(str))
X['region'] = lbl.fit_transform(X['region'].astype(str))

def grid_search(X, y):

	weights = X['factor']
	X = X.drop(['factor'], axis = 1)
	
	lr_rate = parameters['xg_learning_rate_gs']
	n_est = parameters['xg_n_estimators_gs']
	mx_dps = parameters['xg_max_depth_gs']
	scale_weights = parameters['xg_scale_pos_weight_gs']
	model = XGBClassifier(use_label_encoder=False) # avoid annoying messages
	
	param_grid = dict({'learning_rate': lr_rate,
                'n_estimators': n_est,
                'max_depth': mx_dps,
                'scale_pos_weight': scale_weights}
                 )

	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='balanced_accuracy')
	grid_result = grid.fit(X, y, sample_weight = weights)
	# report the best configuration
	print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
	# report all configurations
	means = grid_result.cv_results_['mean_test_score']
	stds = grid_result.cv_results_['std_test_score']
	params = grid_result.cv_results_['params']
	for mean, stdev, param in zip(means, stds, params):
		print("%f (%f) with: %r" % (mean, stdev, param))

	return grid_result.best_params_

print('Initializing grid search...')
# best_params = grid_search(X, y)

# grid search takes hours and my pc cannot handle it. This, using fixed params
best_params = dict({'learning_rate': parameters['xg_learning_rate'],
                'n_estimators': parameters['xg_n_estimators'],
                'max_depth': parameters['xg_max_depth'],
                'scale_pos_weight': parameters['xg_scale_pos_weight']}
                 )

print('Best parameters:')
print(best_params)

print('Initializing train-test split...')
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=parameters['test_size'], random_state=parameters['random_state'])

# Create weights
weights_train = X_train['factor']
weights_test = X_test['factor']

# Drop weights
X_train = X_train.drop(['factor'], axis = 1)
X_test = X_test.drop(['factor'], axis = 1)

# Train model
print('Training the model...')
xg_model = XGBClassifier(**best_params)
xg_model = xg_model.fit(X_train, y_train, sample_weight = weights_train)

# Prediction
print('Predicting...')
y_pred = xg_model.predict(X_test)
y_pred_proba =  xg_model.predict_proba(X_test)

# Evaluate performance
print('Evaluating performance...')
stored_cm = confusion_matrix(y_pred=y_pred, y_true=y_test, labels=[0,1], sample_weight = weights_test)
print(stored_cm)
stored_recall = recall_score(y_pred=y_pred, y_true=y_test, average='binary', sample_weight = weights_test)
print("Recall = {}".format(stored_recall))

## Create a time tag
now = datetime.now()
time_tag = now.strftime("%Y%m%d_%H%M%S")

# Decile performace
df2 = X_test
df2['propension'] = y_pred_proba[:,1]
df2['death'] = y_test
df2['weights'] = weights_test

## weighted deciles
def weighted_qcut(values, weights, q, **kwargs):
    'Return weighted quantile cuts from a given series, values.'
    if is_integer(q):
        quantiles = np.linspace(0, 1, q + 1)
    else:
        quantiles = q
    order = weights.iloc[values.argsort()].cumsum()
    bins = pd.cut(order / order.iloc[-1], quantiles, **kwargs)
    return bins.sort_index()

## unweighted version
# df2['decile'] = pd.qcut(df2['propension'], 10, labels=False)

## weighted version
df2['decile'] = weighted_qcut(df2['propension'], df2['weights'], 20, labels=False)

## weighted average function
wm = lambda x: np.average(x, weights=df2.loc[x.index, "weights"])

analysis_decil = df2.groupby('decile').aggregate({'propension': wm, 'weights' : 'sum', 'death' : 'sum'}).reset_index()

analysis_decil['decile'] = analysis_decil['decile'] + 1
print(analysis_decil)

analysis_decil.to_excel('0_data/8_reporting/' + time_tag + '_xg_deciles.xlsx', index = False)

lines = ['time tag: ' + time_tag, 'used parameters: ' + str(best_params), 'test size: ' + str(parameters['test_size']), 'recall: ' + str(stored_recall), 'confusion matrix: ' + str(stored_cm) ]

with open('0_data/8_reporting/' + time_tag + '_xgb.txt',"w+") as f:
	for line in lines:
		f.write(line)
		f.write('\n')

f.close()
g.close()

if parameters['ft_importance']:
	print('Calculating feature importance...')
	feature_list = list(X_test.columns)
	importances = list(xg_model.feature_importances_)

	df3 = pd.DataFrame(list((zip(feature_list,importances))),
					columns = ['Variable','Importancia relativa'])

	df3.to_excel('0_data/7_model_output/' + time_tag + '_ft_importance.xlsx', index= False)