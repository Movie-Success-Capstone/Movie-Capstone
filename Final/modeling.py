#   ▄▄       ▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄   ▄▄▄▄▄▄▄▄▄▄▄  ▄            ▄▄▄▄▄▄▄▄▄▄▄  ▄▄        ▄  ▄▄▄▄▄▄▄▄▄▄▄ 
#  ▐░░▌     ▐░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░▌ ▐░░░░░░░░░░░▌▐░▌          ▐░░░░░░░░░░░▌▐░░▌      ▐░▌▐░░░░░░░░░░░▌
#  ▐░▌░▌   ▐░▐░▌▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀▀▀ ▐░▌           ▀▀▀▀█░█▀▀▀▀ ▐░▌░▌     ▐░▌▐░█▀▀▀▀▀▀▀▀▀ 
#  ▐░▌▐░▌ ▐░▌▐░▌▐░▌       ▐░▌▐░▌       ▐░▌▐░▌          ▐░▌               ▐░▌     ▐░▌▐░▌    ▐░▌▐░▌          
#  ▐░▌ ▐░▐░▌ ▐░▌▐░▌       ▐░▌▐░▌       ▐░▌▐░█▄▄▄▄▄▄▄▄▄ ▐░▌               ▐░▌     ▐░▌ ▐░▌   ▐░▌▐░▌ ▄▄▄▄▄▄▄▄ 
#  ▐░▌  ▐░▌  ▐░▌▐░▌       ▐░▌▐░▌       ▐░▌▐░░░░░░░░░░░▌▐░▌               ▐░▌     ▐░▌  ▐░▌  ▐░▌▐░▌▐░░░░░░░░▌
#  ▐░▌   ▀   ▐░▌▐░▌       ▐░▌▐░▌       ▐░▌▐░█▀▀▀▀▀▀▀▀▀ ▐░▌               ▐░▌     ▐░▌   ▐░▌ ▐░▌▐░▌ ▀▀▀▀▀▀█░▌
#  ▐░▌       ▐░▌▐░▌       ▐░▌▐░▌       ▐░▌▐░▌          ▐░▌               ▐░▌     ▐░▌    ▐░▌▐░▌▐░▌       ▐░▌
#  ▐░▌       ▐░▌▐░█▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄▄▄ ▐░█▄▄▄▄▄▄▄▄▄  ▄▄▄▄█░█▄▄▄▄ ▐░▌     ▐░▐░▌▐░█▄▄▄▄▄▄▄█░▌
#  ▐░▌       ▐░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░▌ ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░▌      ▐░░▌▐░░░░░░░░░░░▌
#   ▀         ▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀   ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀        ▀▀  ▀▀▀▀▀▀▀▀▀▀▀ 
#
#----------------------------------------------------------------------------------------------------------|
#      I M P O R T S
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wrangle import *
import scipy as sc
import seaborn as sns
import wrangle as w

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.tree import export_text, DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, recall_score
from sklearn.metrics import precision_score, f1_score, accuracy_score, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from matplotlib.ticker import ScalarFormatter
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SequentialFeatureSelector
#----------------------------------------------------------------------------------------------------------|
#----------------------------------------------------------------------------------------------------------|
def create_modeling_df(use_cache=True):
    # If the cached parameter is True, read the csv file on disk in the same folder as this file 
    if os.path.exists('modeling.csv') and use_cache:
        print('modeling.csv detected. \n Dataframe available.')
        return pd.read_csv('modeling.csv')

    # When there's no cached csv, read the following query from Codeup's SQL database.
    print('modeling.csv not detected.')
    print('initiating acquisiton and preparation')
    
    df = w.wrangle_df()
    # obtain ten most frequently occuring companies
    threshold1 = 101
    df.loc[df['production_company'].value_counts()\
           [df['production_company']].values < threshold1, 'production_company'] = "other_company"
    # obtain ten actors who appear the most 
    threshold2 = 26
    df.loc[df['cast_actor_1'].value_counts()\
           [df['cast_actor_1']].values < threshold2, 'cast_actor_1'] = "other_actor"
    # create dummies based on those newly created columns
    dummy_group = ['cast_actor_1', 'production_company',
                  'returns', 'budget_range', 'release_weekday']
    dummy_df = pd.get_dummies(df.loc[:,dummy_group], drop_first=True)
    # subset the data frame. Will retain even less columns after feature selection
    keep =  ['budget','runtime', 'vote_average','vote_count', 'success', 
         'release_year', 'is_genre_adventure', 'is_genre_horror', 
         'is_genre_drama', 'is_genre_scifi', 'is_genre_romance',
         'is_genre_thriller', 'is_genre_crime', 'is_genre_comedy',
         'is_genre_animation', 'is_genre_action', 'is_genre_mystery',
         'is_genre_fantasy', 'is_genre_documentary', 'total_n_cast',
             'is_long_movie', 'ROI']
    modeling_df = df.loc[:,keep]
    modeling_df = pd.concat([modeling_df, dummy_df], axis=1)
    print(f'the current shape is {modeling_df.shape}')
    print('please split and then scale this dataframe')
    
    modeling_df['baseline'] = 0
    modeling_df['baseline_accuracy'] = (modeling_df['baseline'] == modeling_df['success']).mean()

    modeling_df.to_csv('modeling.csv')
    print('modeling.csv ready for future use')
    return modeling_df
#----------------------------------------------------------------------------------------------------------|
#----------------------------------------------------------------------------------------------------------|
def split_and_scale(modeling_df):
    train, validate, test = train_validate_test_split(modeling_df)
    X_train = train.drop(columns=['success'])
    y_train = train['success']
    
    X_validate = validate.drop(columns=['success'])
    y_validate = validate['success']
    
    X_test = test.drop(columns=['success'])
    y_test = test['success']
    
    scaler = MinMaxScaler()
    # Note that we only call .fit with the training data,
    # but we use .transform to apply the scaling to all the data splits.
    scaler.fit(X_train)
    
    X_train_scaled = scaler.transform(X_train)
    X_validate_scaled = scaler.transform(X_validate)
    X_test_scaled = scaler.transform(X_test)
    
    X_train = pd.DataFrame(X_train_scaled, index=X_train.index, columns = X_train.columns)
    X_validate = pd.DataFrame(X_validate_scaled, index=X_validate.index, columns = X_validate.columns)
    X_test = pd.DataFrame(X_test_scaled, index=X_test.index, columns = X_test.columns)
    
    return X_train, X_validate, X_test, y_train, y_validate, y_test
#----------------------------------------------------------------------------------------------------------|
#----------------------------------------------------------------------------------------------------------|

def print_cv_results(gs, title):
    print('\n -----------------------------------------')
    print(title)

    print(f'Best Score = {gs.best_score_:.4f}')
    print(f'Best Hyper-parameters = {gs.best_params_}')
    print()

    print('Test Scores:')
    test_means = gs.cv_results_['mean_test_score']
    test_stds = gs.cv_results_['std_test_score']
    for mean, std, params in zip(test_means, test_stds, gs.cv_results_['params']):
        print(f'{mean:.4f} (+/-{std:.3f}) for {params}')
    print()

    print('Training Scores:')
    train_means = gs.cv_results_['mean_train_score']
    train_stds = gs.cv_results_['std_train_score']
    for mean, std, params in zip(train_means, train_stds, gs.cv_results_['params']):
        print(f'{mean:.4f} (+/-{std:.3f}) for {params}')
    print('\n -----------------------------------------')   
#----------------------------------------------------------------------------------------------------------|
#----------------------------------------------------------------------------------------------------------|
def accuracy_models(X_train, y_train):
    # Logistic Regression
    logReg = LogisticRegression(max_iter=1000)
    c_list = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    param_grid = {'C': c_list,
                  'solver': ['newton-cg', 'lbfgs', 'sag', 'saga']}
    
    gs = GridSearchCV(estimator=logReg,
                      param_grid=param_grid,
                      scoring='accuracy',
                      cv=3,
                      return_train_score=True)
    gs = gs.fit(X_train, y_train)
    print_cv_results(gs, 'Logistic Regression Accuracy')
    print('------------------------------------------------')
    # KNN
    knn = KNeighborsClassifier()
    k_list = list(range(1, 26, 2))
    param_grid = [{'n_neighbors': k_list}]
    
    gs = GridSearchCV(estimator=knn,
                      param_grid=param_grid,
                      scoring='accuracy',
                      cv=3,
                      return_train_score=True)
    gs = gs.fit(X_train, y_train)
    print_cv_results(gs, 'KNN Accuracy')
    
    
    test_means = gs.cv_results_['mean_test_score']
    train_means = gs.cv_results_['mean_train_score']
    
    plt.plot(k_list, test_means, marker='o', label='Test')
    plt.plot(k_list, train_means, marker='o', label='Train')
    plt.xticks(k_list)
    
    plt.title('Movie Success Prediction: KNN')
    plt.ylabel('Accuracy')
    plt.xlabel('Number of Neighbors')
    plt.legend()
    plt.show()
    print('------------------------------------------------')
    
    # Decision Tree
    criterion = ['gini', 'entropy']
    colors = ['red', 'blue']
    depth_list = list(range(1,17))
    
    for i in range(len(criterion)):
        tree = DecisionTreeClassifier(criterion=criterion[i],
                                     splitter='best')
        param_grid = [{'max_depth': depth_list}]
        gs = GridSearchCV(estimator=tree,
                          param_grid=param_grid,
                          scoring='accuracy',
                          cv=5,
                         return_train_score=True)
        gs = gs.fit(X_train, y_train)
        print_cv_results(gs, 'Decision Tree Regression Accuracy')
    
        test_means = gs.cv_results_['mean_test_score']
        train_means = gs.cv_results_['mean_train_score']
    
        plt.plot(depth_list, test_means, marker='o', label=f'{criterion[i]} Test Mean',
                    color=colors[i])
        plt.plot(depth_list, train_means, marker='o', label=f'{criterion[i]} Train Mean',
                    linestyle='dashed', color=colors[i])
    
    plt.xticks(depth_list)
    plt.title(f'Movie Success Prediction: Decision Tree')
    plt.ylabel('Accuracy')
    plt.xlabel('Max Tree Depth')
    plt.legend()
    plt.show()
    print('------------------------------------------------')
    # Random Forest   
    forest = RandomForestClassifier()
    criterion = ['gini', 'entropy']
    n_list = list(range(1, 11))
    param_grid = [{'n_estimators': n_list,
                    'max_depth': n_list,
                    'criterion': criterion}]
    gs = GridSearchCV(estimator=forest,
                      param_grid=param_grid,
                      scoring='accuracy',
                      cv=5,
                     return_train_score=True)
    gs = gs.fit(X_train, y_train)
    
    
    # print line graph of random forest where max_depth=8
    criterion = ['gini', 'entropy']
    colors = ['red', 'blue']
    n_list = list(range(1, 11))
    for i in range(len(criterion)):
        forest = RandomForestClassifier(criterion=criterion[i], max_depth=8)
        param_grid = [{'n_estimators': n_list}]
        gs = GridSearchCV(estimator=forest,
                          param_grid=param_grid,
                          scoring='accuracy',
                          cv=5,
                         return_train_score=True)
        gs = gs.fit(X_train, y_train)
        print_cv_results(gs, 'Random Forest Accuracy')
    
        test_means = gs.cv_results_['mean_test_score']
        train_means = gs.cv_results_['mean_train_score']
    
        plt.plot(n_list, test_means, marker='o', label=f'{criterion[i]} Test Mean',
                    color=colors[i])
        plt.plot(n_list, train_means, marker='o', label=f'{criterion[i]} Train Mean',
                    linestyle='dotted', color=colors[i])
    
    plt.xticks(n_list)
    plt.title(f'Movie Success Prediction: Random Forest, max_depth=10')
    plt.ylabel('Accuracy')
    plt.xlabel('Number of Trees')
    plt.legend()
    plt.show()
    print('------------------------------------------------')
    print('------------------------------------------------')
#----------------------------------------------------------------------------------------------------------|
#----------------------------------------------------------------------------------------------------------|
def precision_models(X_train, y_train):

    # Logistic Regression
    logReg = LogisticRegression(max_iter=1000)
    c_list = [ 1, 10, 100, 1000]
    param_grid = {'C': c_list,
                  'solver': ['newton-cg', 'lbfgs', 'sag', 'saga']}
    
    gs = GridSearchCV(estimator=logReg,
                      param_grid=param_grid,
                      scoring = 'precision',
                      cv=3,
                      return_train_score=True)
    gs = gs.fit(X_train, y_train)
    print_cv_results(gs, 'Logistic Regression Precision')
    print('------------------------------------------------')
    # KNN
    knn = KNeighborsClassifier()
    k_list = list(range(1, 26, 2))
    param_grid = [{'n_neighbors': k_list}]
    
    gs = GridSearchCV(estimator=knn,
                      param_grid=param_grid,
                      scoring='precision',
                      cv=3,
                      return_train_score=True)
    gs = gs.fit(X_train, y_train)
    print_cv_results(gs, 'KNN Precision')
    
    
    test_means = gs.cv_results_['mean_test_score']
    train_means = gs.cv_results_['mean_train_score']
    
    plt.plot(k_list, test_means, marker='o', label='Test')
    plt.plot(k_list, train_means, marker='o', label='Train')
    plt.xticks(k_list)
    
    plt.title('Movie Success Prediction: KNN')
    plt.ylabel('Precision')
    plt.xlabel('Number of Neighbors')
    plt.legend()
    plt.show()
    print('------------------------------------------------')
    # Decision Tree
    criterion = ['gini', 'entropy']
    colors = ['red', 'blue']
    depth_list = list(range(1,17))
    
    for i in range(len(criterion)):
        tree = DecisionTreeClassifier(criterion=criterion[i],
                                     splitter='best')
        param_grid = [{'max_depth': depth_list}]
        gs = GridSearchCV(estimator=tree,
                          param_grid=param_grid,
                          scoring='precision',
                          cv=5,
                         return_train_score=True)
        gs = gs.fit(X_train, y_train)
        print_cv_results(gs, 'Decision Tree Regression Precision')
    
        test_means = gs.cv_results_['mean_test_score']
        train_means = gs.cv_results_['mean_train_score']
    
        plt.plot(depth_list, test_means, marker='o', label=f'{criterion[i]} Test Mean',
                    color=colors[i])
        plt.plot(depth_list, train_means, marker='o', label=f'{criterion[i]} Train Mean',
                    linestyle='dashed', color=colors[i])
    
    plt.xticks(depth_list)
    plt.title(f'Movie Success Prediction: Decision Tree')
    plt.ylabel('Precision')
    plt.xlabel('Max Tree Depth')
    plt.legend()
    plt.show()
    print('------------------------------------------------')
    print('------------------------------------------------')
#----------------------------------------------------------------------------------------------------------|
#----------------------------------------------------------------------------------------------------------|
def recall_models(X_train, y_train):
    # Logistic Regression
    logReg = LogisticRegression(max_iter=10000)
    c_list = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
    param_grid = {'C': c_list,
                  'solver': ['newton-cg', 'lbfgs', 'sag', 'saga']}
    
    gs = GridSearchCV(estimator=logReg,
                      param_grid=param_grid,
                      scoring='recall',
                      cv=3,
                      return_train_score=True)
    gs = gs.fit(X_train, y_train)
    print_cv_results(gs, 'Logistic Regression Recall')
    print('------------------------------------------------')
    # KNN
    knn = KNeighborsClassifier()
    k_list = list(range(1, 26, 2))
    param_grid = [{'n_neighbors': k_list}]
    
    gs = GridSearchCV(estimator=knn,
                      param_grid=param_grid,
                      scoring='recall',
                      cv=3,
                      return_train_score=True)
    gs = gs.fit(X_train, y_train)
    print_cv_results(gs, 'KNN Recall')
    
    
    test_means = gs.cv_results_['mean_test_score']
    train_means = gs.cv_results_['mean_train_score']
    
    plt.plot(k_list, test_means, marker='o', label='Test')
    plt.plot(k_list, train_means, marker='o', label='Train')
    plt.xticks(k_list)
    
    plt.title('Movie Success Prediction: KNN')
    plt.ylabel('Recall')
    plt.xlabel('Number of Neighbors')
    plt.legend()
    plt.show()
    print('------------------------------------------------')
    # Decision Tree
    criterion = ['gini', 'entropy']
    colors = ['red', 'blue']
    depth_list = list(range(1,17))
    
    for i in range(len(criterion)):
        tree = DecisionTreeClassifier(criterion=criterion[i],
                                     splitter='best')
        param_grid = [{'max_depth': depth_list}]
        gs = GridSearchCV(estimator=tree,
                          param_grid=param_grid,
                          scoring='recall',
                          cv=5,
                         return_train_score=True)
        gs = gs.fit(X_train, y_train)
        print_cv_results(gs, 'Decision Tree Regression Recall')
    
        test_means = gs.cv_results_['mean_test_score']
        train_means = gs.cv_results_['mean_train_score']
    
        plt.plot(depth_list, test_means, marker='o', label=f'{criterion[i]} Test Mean',
                    color=colors[i])
        plt.plot(depth_list, train_means, marker='o', label=f'{criterion[i]} Train Mean',
                    linestyle='dashed', color=colors[i])
    
    plt.xticks(depth_list)
    plt.title(f'Movie Success Prediction: Decision Tree')
    plt.ylabel('Recall')
    plt.xlabel('Max Tree Depth')
    plt.legend()
    plt.show()
    print('------------------------------------------------')
    # Random Forest
    forest = RandomForestClassifier()
    criterion = ['gini', 'entropy']
    n_list = list(range(1, 11))
    param_grid = [{'n_estimators': n_list,
                    'max_depth': n_list,
                    'criterion': criterion}]
    gs = GridSearchCV(estimator=forest,
                      param_grid=param_grid,
                      scoring='recall',
                      cv=5,
                     return_train_score=True)
    gs = gs.fit(X_train, y_train)
    
    
    # print line graph of random forest where max_depth=8
    criterion = ['gini', 'entropy']
    colors = ['red', 'blue']
    n_list = list(range(1, 11))
    for i in range(len(criterion)):
        forest = RandomForestClassifier(criterion=criterion[i], max_depth=8)
        param_grid = [{'n_estimators': n_list}]
        gs = GridSearchCV(estimator=forest,
                          param_grid=param_grid,
                          scoring='recall',
                          cv=5,
                         return_train_score=True)
        gs = gs.fit(X_train, y_train)
        print_cv_results(gs, 'Random Forest Recall')
    
        test_means = gs.cv_results_['mean_test_score']
        train_means = gs.cv_results_['mean_train_score']
    
        plt.plot(n_list, test_means, marker='o', label=f'{criterion[i]} Test Mean',
                    color=colors[i])
        plt.plot(n_list, train_means, marker='o', label=f'{criterion[i]} Train Mean',
                    linestyle='dotted', color=colors[i])
    
    plt.xticks(n_list)
    plt.title(f'Movie Success Prediction: Random Forest, max_depth=10')
    plt.ylabel('Recall')
    plt.xlabel('Number of Trees')
    plt.legend()
    plt.show()
#----------------------------------------------------------------------------------------------------------|
#----------------------------------------------------------------------------------------------------------|
def predict_on_test(X_train, y_train, X_test, y_test):
    lr = LogisticRegression(max_iter=10000,
                        C = 100,
                        solver ='newton-cg',
                        multi_class='auto')

    # prediction_accuracies.
    prediction_accuracies = []
    confusion_matrices = []
    
    # predictions.
    model_predictions = [] 
    target_predictions = []
    
    # Train the model using training sets.
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    prediction_accuracies.append(accuracy)
    # Metrics.
    confusion_matrices.append(metrics.confusion_matrix(y_test, y_pred))
    
    # Predictions and Targets.
    target_predictions.append(y_test)
    model_predictions.append(y_pred)
    
    totals = np.array([[0, 0],[0, 0]])
    
    for matrix in confusion_matrices:
    
        for i in range(0, 2):
            for j in range(0, 2):
                totals[i][j] += matrix[i][j]
    
    tn, fp, fn, tp = totals.ravel()
    
    cm = np.array([['tn: '+str(tn), 'fp: '+str(fp)],['fn: '+str(fn), 'tp: '+str(tp)]])
    
    print("------------------")
    total_records = tn+fp+fn+tp
    print("Number of records: ", total_records)
    print("------------------")
    print("Confusion Matrix:")
    print(cm)
    print("------------------")
    avg = sum(prediction_accuracies)/len(prediction_accuracies)
    print("Accuracy: %.9f %%" %(avg*100))
    print("------------------")
    tpr = (tp/(tp+fn))*100
    tnr = (tn/(tn+fp))*100
    fpr = (fp/(tn+fp))*100
    fnr = (fn/(tp+fn))*100
    print("TPR: %.4f %%" %(tpr))
    print("TNR: %.4f %%" %(tnr))
    print("FPR: %.4f %%" %(fpr))
    print("FNR: %.4f %%" %(fnr))
    print("------------------")
    precision = (tp/(tp+fp))
    recall = (tp/(tp+fn))
    print("Precision: %.4f" %(precision))
    print("Recall: %.4f" %(recall))
    print("------------------")
    f1measure = 2*((precision*recall)/(precision+recall))
    print("F1-Measure: %.4f" %(f1measure))
    
    
    # Predictions
    target_predictions = np.concatenate((target_predictions), axis=None)
    model_predictions = np.concatenate((model_predictions), axis=None)
    
    
    fpr, tpr, thresholds = roc_curve(model_predictions, target_predictions)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    lw=2
    plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Logistic Regression ROC Curve.png')
    plt.show()
#----------------------------------------------------------------------------------------------------------|
#----------------------------------------------------------------------------------------------------------|
#----------------------------------------------------------------------------------------------------------|