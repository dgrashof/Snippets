import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics


def data_proc(df,cat_fields,cont_fields,response):
    df = df
    cat_fields = cat_fields
    cont_fields = cont_fields
    response = response
    # Split data into train and test set
    _Xtrain, _Xtest, ytrain, ytest=train_test_split(
        df[cat_fields+cont_fields],df[response], test_size=.3, random_state=42)
    
    # Impute missing categorical or continuous values
    catimp = SimpleImputer(missing_values="", strategy="most_frequent")
    contimp = SimpleImputer(missing_values=np.NaN, strategy="most_frequent")
    # Apply imputers to training data
    _cont_df, _cat_df = _Xtrain[cont_fields], _Xtrain[cat_fields]
    cont_df_imp = contimp.fit_transform(_cont_df)
    cat_df_imp = catimp.fit_transform(_cat_df.astype(str))
    cat_df_imp_ = pd.DataFrame(cat_df_imp, columns=_cat_df.columns)
    # Scale continuous features to eliminate magnitude bias - training
    std_scaler = StandardScaler()
    cont_df_scl = pd.DataFrame(
        std_scaler.fit_transform(cont_df_imp), columns=_cont_df.columns)
    # Recombine cat and continuous fields - train
    Xtrain = pd.concat([cont_df_scl, cat_df_imp_], axis=1)
    # Apply imputers to test data
    _cont_df2, _cat_df2 = _Xtest[cont_fields], _Xtest[cat_fields]
    cont_df_imp2 = contimp.transform(_cont_df2)
    cat_df_imp2 = catimp.transform(_cat_df2.astype(str))
    cat_df_imp2 = pd.DataFrame(cat_df_imp2, columns=_cat_df2.columns)
    # Scale continuous features to eliminate magnitude bias. - test
    cont_df_scl2 = pd.DataFrame(
        std_scaler.transform(cont_df_imp2), columns=_cont_df.columns)
    # Recombine cat and continuous fields - test
    Xtest = pd.concat([cont_df_scl2, cat_df_imp2], axis=1)
    # Save scaler for model deployment else comment out
    pickle.dump(std_scaler, open('scaler.sav', 'wb'))
    
    # One hot encode categorical variables as combined test and train dataset
    # Define variable to identify test and train data
    Xtrain["train"] = 1
    Xtest["train"] = 0
    # Combine test and train data
    combined = pd.concat([Xtrain, Xtest])
    # isolate categorical variables
    cat_df = combined[cat_fields]
    # create encoder and fit; ignore unknown values that come up in predict data set
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(cat_df)
    # transform data set using encoder
    onehotlabels = enc.transform(cat_df).toarray()
    # create df of one-hot array with original labels from df
    cat_df_enc = pd.DataFrame(onehotlabels, columns=enc.get_feature_names(input_features=cat_df.columns))
    # Combine cat/cont variables
    combined.reset_index(inplace=True, drop=True)
    combined_2 = pd.concat([cat_df_enc, combined[cont_fields], combined['train']], axis=1)
    # split combined data set back to train and test
    Xtrain = combined_2[combined_2["train"] == 1]
    Xtest = combined_2[combined_2["train"] == 0]
    Xtrain.drop(["train"], axis=1, inplace=True)
    Xtest.drop(["train"], axis=1, inplace=True)
    # save encoder for predict data set
    pickle.dump(enc, open('encoder.sav', 'wb'))
    
    # Deploy SMOTE algorithm in case data set is unbalanced
    sm = SMOTE(sampling_strategy='auto')
    X_resampled, y_resampled = sm.fit_sample(Xtrain, ytrain)
    Xtrain = pd.DataFrame(X_resampled, columns=Xtrain.columns)
    ytrain = pd.Series(y_resampled)
    
    # Use feature selection function to calculate the optimal number of features. Only applicable classification
    # problems
    def feature_selection(xtrain,ytrain,xtest,ytest):
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
            thresh_df = thresh_df.append({'thresh':thresh, 'num':select_xtrain.shape[1], 'recall':recall},ignore_index=True)
        thresh_df=thresh_df[thresh_df['num']!=1.0]
        val = thresh_df['num'].values[thresh_df['recall'] == thresh_df['recall'].max()].max().astype(int)
        feature_matrix = pd.DataFrame({'field':Xtest.columns,'value':init_model.feature_importances_}).sort_values(by = 'value',ascending = False)
        cols = feature_matrix['field'][:val].values.tolist()
        xtrain = xtrain[cols]
        xtest = xtest[cols]
        return(xtrain,xtest)
    
    Xtrain,Xtest = feature_selection(xtrain=Xtrain,ytrain=ytrain,xtest=Xtest,ytest=ytest)
    return(Xtrain,ytrain,Xtest,ytest)