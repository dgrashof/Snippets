def feature_selection(xtrain,ytrain,xtest,ytest):
	from xgboost import XGBClassifier
	from sklearn.linear_model import LogisticRegression
	from sklearn.feature_selection import SelectFromModel
	import pandas as pd
	init_model = XGBClassifier(random_state = 42)
	init_model.fit(xtrain,ytrain)
	# make predictions for test data and evaluate
	thresholds = np.sort(np.unique(init_model.feature_importances_))
	thresh_df = pd.DataFrame()
	for thresh in thresholds:
		# select features using threshold
		selection = SelectFromModel(init_model, threshold=thresh, prefit=True)
		select_xtrain = selection.transform(xtrain)
		# train model
		selection_model = LogisticRegression(max_iter = 1000)
		selection_model.fit(select_xtrain, ytrain)
		# eval model
		select_xtest = selection.transform(xtest)
		y_pred = selection_model.predict(select_xtest)
		predictions = [round(value) for value in y_pred]
		recall = metrics.recall_score(ytest, predictions)
		#create dataframe to store results
		thresh_df = thresh_df.append({'thresh':thresh, 'num':select_xtrain.shape[1], 'recall':recall},ignore_index=True)
	#pick threshold with highest recall, excluding using only one feature
	thresh_df=thresh_df[thresh_df['num']!=1.0]
	val = thresh_df['num'].values[thresh_df['recall'] == thresh_df['recall'].max()].max().astype(int)
	feature_matrix = pd.DataFrame({'field':Xtest.columns,'value':init_model.feature_importances_}).sort_values(by = 'value',ascending = False)
	cols = feature_matrix['field'][:val].values.tolist()
	#filter train and test data to only include desired features
	xtrain = xtrain[cols]
	xtest = xtest[cols]
	return(xtrain,xtest)