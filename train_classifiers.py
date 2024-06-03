import pandas as pd
from label_feature_vectors import NEW_FEATURE_VECTOR_FILENAME
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

SEED = 42
VALIDATION_SIZE = 0.2
MAX_ITERS = [100, 500, 1000, 1500, 2000]

MODEL_HYPERPARAMS = {
    'SVC': {
        'C': [1, 5, 10, 20, 50],
        'kernel': ['linear', 'rbf'],
        'gamma': ['auto']
    },
    'GaussianNB': {},  # No hyperparams needed
    'DecisionTreeClassifier': {
        'criterion': [
            'gini',
            'entropy',
            'log_loss'
        ],
        'splitter': ['best', 'random'],
        'max_depth': [None, 10, 20, 30, 40]
    },
    'MLPClassifier': {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],
        'activation': ['relu', 'tanh', 'logistic'],
        'solver': ['adam', 'sgd'],
        'learning_rate': ['constant', 'adaptive'],
        'learning_rate_init': [0.0001, 0.0005, 0.001, 0.005, 0.01]
    },
    'RandomForestClassifier': {  # Only n_estimators different from Decision tree, rest same
        'n_estimators': [10, 20, 40, 80],
        'criterion': [
            'gini',
            'entropy',
            'log_loss'
        ],
        'max_depth': [None, 10, 20, 30, 40]
    }
}


def train_SVC(X_train, y_train, X_val, y_val, hyperparams):
    print('Fitting SVC Models')
    best_model = None
    history = list()
    for c in hyperparams['C']:
        for kernel in hyperparams['kernel']:
            for gamma in hyperparams['gamma']:
                model = SVC(C=c, kernel=kernel, gamma=gamma)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                precision, recall, _, _ = precision_recall_fscore_support(y_val, y_pred, average='binary',
                                                                          zero_division=0.0)
                f1score = f1_score(y_val, y_pred)
                history.append({
                    'params': {
                        'C': c,
                        'kernel': kernel,
                        'gamma': gamma
                    },
                    'precision': precision,
                    'recall': recall,
                    'f1': f1score
                })
                if best_model is None or best_model[1] + best_model[2] < precision + recall:
                    best_model = (model, precision, recall)
                    print(f"Best Model is with params: C = {c}, kernel = {kernel}, gamma = {gamma}")
    return best_model[0], history


def train_NB(X_train, y_train, X_val, y_val, hyperparams):
    print("Fitting NB Model")
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    precision, recall, _, _ = precision_recall_fscore_support(y_val, y_pred, average='binary', zero_division=0.0)
    f1score = f1_score(y_val, y_pred)
    history = [{
        'params': {
        },
        'precision': precision,
        'recall': recall,
        'f1': f1score
    }]
    return model, history


def train_MLP(X_train, y_train, X_val, y_val, hyperparams):
    print("Fitting MLP Models")
    best_model = None
    history = list()
    for max_iters in MAX_ITERS:
        for learning_rate_init in hyperparams['learning_rate_init']:
            for learning_rate in hyperparams['learning_rate']:
                for hidden_layer_sizes in hyperparams['hidden_layer_sizes']:
                    for activation in hyperparams['activation']:
                        for solver in hyperparams['solver']:
                            model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation,
                                                  solver=solver, learning_rate=learning_rate,
                                                  learning_rate_init=learning_rate_init, max_iter=max_iters)
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_val)
                            precision, recall, _, _ = precision_recall_fscore_support(y_val, y_pred, average='binary',
                                                                                      zero_division=0.0)
                            f1score = f1_score(y_val, y_pred)
                            history.append({
                                'params': {
                                    'hidden_layer_sizes': hidden_layer_sizes,
                                    'activation': activation,
                                    'solver': solver,
                                    'learning_rate': learning_rate,
                                    'learning_rate_init': learning_rate_init
                                },
                                'precision': precision,
                                'recall': recall,
                                'f1': f1score
                            })
                            if best_model is None or best_model[1] + best_model[2] < precision + recall:
                                best_model = (model, precision, recall)
                                print(
                                    f"Best Model is with params: Hidden Layer Size = {hidden_layer_sizes}, Learning "
                                    f"Rate = {learning_rate}, Learning Rate Init = {learning_rate_init}, Activation = "
                                    f"{activation}, Max Iters = {max_iters}, Solver = {solver}")
    return best_model[0], history


def train_RFC(X_train, y_train, X_val, y_val, hyperparams):
    print("Fitting RFC Models")
    best_model = None
    history = list()
    for n_estimators in hyperparams['n_estimators']:
        for criterion in hyperparams['criterion']:
            for max_depth in hyperparams['max_depth']:
                model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                precision, recall, _, _ = precision_recall_fscore_support(y_val, y_pred, average='binary',
                                                                          zero_division=0.0)
                f1score = f1_score(y_val, y_pred)
                history.append({
                    'params': {
                        'criterion': criterion,
                        'n_estimators': n_estimators,
                        'max_depth': max_depth
                    },
                    'precision': precision,
                    'recall': recall,
                    'f1': f1score
                })
                if best_model is None or best_model[1] + best_model[2] < precision + recall:
                    best_model = (model, precision, recall)
                    print(
                        f"Best Model is with params: N_estimators = {n_estimators}, Criterion = {criterion}, Max Depth = {max_depth}")
    return best_model[0], history


def train_decision_tree(X_train, y_train, X_val, y_val, hyperparams):
    print("Fitting Decision Tree Models")
    best_model = None
    history = list()
    for splitter in hyperparams['splitter']:
        for criterion in hyperparams['criterion']:
            for max_depth in hyperparams['max_depth']:
                model = DecisionTreeClassifier(splitter=splitter, criterion=criterion, max_depth=max_depth)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                precision, recall, _, _ = precision_recall_fscore_support(y_val, y_pred, average='binary',
                                                                          zero_division=0.0)
                f1score = f1_score(y_val, y_pred)
                history.append({
                    'params': {
                        'criterion': criterion,
                        'splitter': splitter,
                        'max_depth': max_depth
                    },
                    'precision': precision,
                    'recall': recall,
                    'f1': f1score
                })
                if best_model is None or best_model[1] + best_model[2] < precision + recall:
                    best_model = (model, precision, recall)
                    print(
                        f"Best Model is with params: Splitter = {splitter}, Criterion = {criterion}, Max Depth = {max_depth}")
    return best_model[0], history


def train_models(X_train, X_valid, y_train, y_valid):
    best_models = dict()  # Stores the best model for each of the classifiers given the hyper parameters
    history = dict()
    for model_name, hyperparams in MODEL_HYPERPARAMS.items():
        match (model_name):
            case "SVC":
                res = train_SVC(X_train, y_train, X_valid, y_valid, hyperparams)
                best_models[model_name] = res[0]
                history[model_name] = res[1]
            case "MLPClassifier":
                res = train_MLP(X_train, y_train, X_valid, y_valid, hyperparams)
                best_models[model_name] = res[0]
                history[model_name] = res[1]
            case "RandomForestClassifier":
                res = train_RFC(X_train, y_train, X_valid, y_valid, hyperparams)
                best_models[model_name] = res[0]
                history[model_name] = res[1]
            case "GaussianNB":
                res = train_NB(X_train, y_train, X_valid, y_valid, hyperparams)
                best_models[model_name] = res[0]
                history[model_name] = res[1]
            case "DecisionTreeClassifier":
                res = train_decision_tree(X_train, y_train, X_valid, y_valid, hyperparams)
                best_models[model_name] = res[0]
                history[model_name] = res[1]
            case _:
                print("No Model Matched. Aborting")
                exit(1)
    return best_models, history  # Gives the best model, precision, and recall score


if __name__ == '__main__':
    feature_vectors = pd.read_csv(NEW_FEATURE_VECTOR_FILENAME)
    features = feature_vectors.drop(columns=['class', 'buggy'])
    target = feature_vectors['buggy']

    X_train, X_valid, y_train, y_valid = train_test_split(features, target, test_size=0.2, random_state=SEED)

    best_models, history = train_models(X_train, X_valid, y_train, y_valid)

