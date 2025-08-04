import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pandas.plotting import parallel_coordinates
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics


def poison_data(X, y, percentage, random_state=42):
    np.random.seed(random_state)
    n_samples = int(len(X) * percentage / 100)
    indices = np.random.choice(len(X), size=n_samples, replace=False)

    X_poisoned = X.copy()
    # Replace selected rows with random values in feature range
    for col in X.columns:
        min_val = X[col].min()
        max_val = X[col].max()
        X_poisoned.iloc[indices, X.columns.get_loc(col)] = np.random.uniform(min_val, max_val, size=n_samples)

    
    return X_poisoned, y

def evaluate_poisoning(percentage):
    X_poisoned, y_poisoned = poison_data(X_train, y_train, percentage)
    
    poisoned_model = DecisionTreeClassifier(max_depth=3, random_state=1)
    poisoned_model.fit(X_poisoned, y_poisoned)
    
    pred = poisoned_model.predict(X_test)
    acc = metrics.accuracy_score(pred, y_test)
    print(f'Accuracy with {percentage}% poisoned data: {acc:.3f}')
    return acc

data = pd.read_csv('data/iris.csv')

train, test = train_test_split(data, test_size = 0.4, stratify = data['species'], random_state = 42)
X_train = train[['sepal_length','sepal_width','petal_length','petal_width']]
y_train = train.species
X_test = test[['sepal_length','sepal_width','petal_length','petal_width']]
y_test = test.species

evaluate_poisoning(0)
evaluate_poisoning(5)
evaluate_poisoning(10)
evaluate_poisoning(50)