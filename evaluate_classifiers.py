import pandas as pd
from label_feature_vectors import NEW_FEATURE_VECTOR_FILENAME
from sklearn.model_selection import train_test_split, cross_validate, RepeatedKFold
from train_classifiers import train_models, SEED
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

SCORING = {'f1': 'f1', 'precision': 'precision', 'recall': 'recall'}


# Define the biased classifier
class BiasedClassifier(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        pass

    def predict(self, X):
        return np.ones(X.shape[0])  # Always predict 1


def evaluate_classifiers(classifiers, features, targets):
    results = {}

    for model_name, model in classifiers.items():
        if model_name == "MLP":
            clf = make_pipeline(StandardScaler(), model)
        else:
            clf = model

        cross_validation_results = cross_validate(clf, features, targets, scoring=SCORING,
                                                  cv=RepeatedKFold(n_splits=5, n_repeats=20, random_state=SEED))
        results[model_name] = {
            'f1': cross_validation_results['test_f1'],
            'precision': cross_validation_results['test_precision'],
            'recall': cross_validation_results['test_recall'],
        }
        print(f"Cross-Validated for {model_name}")

    return results


def append_biased_score_to_cv_results(features, target, cv_res):
    scoring = {'f1': 'f1', 'precision': 'precision', 'recall': 'recall'}
    cv = RepeatedKFold(n_splits=5, n_repeats=20, random_state=42)
    results = cross_validate(BiasedClassifier(), features, target, scoring=SCORING, cv=cv)
    cv_res["Biased_Model"] = {
        'f1': results['test_f1'],
        'precision': results['test_precision'],
        'recall': results['test_recall'],
    }


def plot_boxplots(scores_dict, metric):
    plt.figure(figsize=(10, 6))
    data = [scores_dict[model_name][metric] for model_name in scores_dict.keys()]
    plt.boxplot(data, labels=scores_dict.keys())
    plt.title(f'Boxplot of {metric} scores for each classifier')
    plt.xlabel('Classifier')
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.grid(True)

    # Add mean and standard deviation to the plot
    for i, model_name in enumerate(scores_dict.keys()):
        mean = np.mean(scores_dict[model_name][metric])
        std = np.std(scores_dict[model_name][metric])
        plt.text(i + 1, mean, f'{mean:.2f}\n±{std:.2f}', ha='center', va='center', color='red')

    plt.show()


def wilcoxon_test(cv_results):
    results = {}

    for model_name1, metrics1 in cv_results.items():
        for model_name2, metrics2 in cv_results.items():
            if model_name1 != model_name2:
                _, p_value_f1 = wilcoxon(metrics1['f1'], metrics2['f1'])
                _, p_value_precision = wilcoxon(metrics1['precision'], metrics2['precision'])
                _, p_value_recall = wilcoxon(metrics1['recall'], metrics2['recall'])

                results[(model_name1, model_name2)] = {
                    'p_value_f1': p_value_f1,
                    'p_value_precision': p_value_precision,
                    'p_value_recall': p_value_recall
                }

    return results


if __name__ == '__main__':
    feature_vectors = pd.read_csv(NEW_FEATURE_VECTOR_FILENAME)
    features = feature_vectors.drop(columns=['class', 'buggy'])
    target = feature_vectors['buggy']

    X_train, X_valid, y_train, y_valid = train_test_split(features, target, test_size=0.2, random_state=SEED)

    best_models, history = train_models(X_train, X_valid, y_train, y_valid)
    cv_res = evaluate_classifiers(best_models, features, target)
    append_biased_score_to_cv_results(features, target, cv_res)
    for k, v in SCORING.items():
        plot_boxplots(cv_res, k)

    cv_comp = dict()

    for model, metrics in cv_res.items():
        cv_comp[model] = dict()
        for metric, values in metrics.items():
            cv_comp[model][metric] = np.mean(values)

    print(cv_comp)
    wox_test_res = wilcoxon_test(cv_res)
    print(cv_res)

