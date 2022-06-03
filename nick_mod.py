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
    