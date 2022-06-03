from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import export_text
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from matplotlib.ticker import ScalarFormatter
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SequentialFeatureSelector
from pre_processing_nick import print_cv_results
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc
import numpy as np

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
    # # KNN
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
    # ############################################################
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
    # # Random Forest
    # get results for random forest
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
    # # KNN
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
    
    # # KNN
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
    
    # ############################################################
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
    
    # # Random Forest
    # get results for random forest
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