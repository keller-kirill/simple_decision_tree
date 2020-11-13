import numpy as np
import pandas as pd
from collections import Counter


def entropy(a_list):
    '''Calculates the Shannon's entropy'''
    c = Counter(a_list)
    entropy = 0
    for cnt in c.values():
        p = cnt/len(a_list)
        entropy -= p*np.log2(p)
        
    return entropy

# information gain calculation
def information_gain(root, left, right):
    '''root - initial data, left and right - two partitions'''
    S0 = entropy(root)
    S_left = entropy(left)
    S_right = entropy(right)
    IG = S0 - (len(left) / len(root) * S_left + len(right) / len(root) * S_right) 
    return IG


def find_split(X, y, feature):
    '''Finds the split by simply averaging average feature values for each class'''
    return np.mean((X[feature][y==1].mean(),X[feature][y==0].mean()))


def split_tree(X, y, feature):
    '''Splits whole tree by split threshold'''
    split = find_split(X, y, feature)
    X_left = X[X[feature] <= split]
    y_left = y[X[feature] <= split]
    X_right = X[X[feature] > split]
    y_right = y[X[feature] > split]
    return X_left, y_left, X_right, y_right


def best_feature_to_split(X, y):
    '''Outputs information gain when splitting on best feature.
    Assumed X to be a DataFrame, y – some vector with the lenght of X.
    y is binary, either 1 or 0
    '''
    gains = {}
    for feature in X.columns:
        split = find_split(X, y, feature)
        left = y[X[feature] <= split]
        right = y[X[feature] > split]
        gain = information_gain(y, left, right)
        gains[feature] = gain
        
    return max(gains, key=gains.get)


def build_tree(X, y):
    '''Builds a decision tree for a dataframe X with labels y'''
    best = best_feature_to_split(X, y)
    split = find_split(X, y, best)
    print('{} >= {:.1f}\n'.format(best, split))
    X_left, y_left, X_right, y_right = split_tree(X, y, best)
    
    entropy_left = entropy(y_left)
    entropy_right = entropy(y_right)
    
    samples_left = len(y_left)
    samples_right = len(y_right)
    
    values_left = np.bincount(y_left)
    class_left = np.argmax(values_left)
    values_right = np.bincount(y_right)
    class_right = np.argmax(values_right)
    
    print('Samples in left {:d}\tSamples in right {:d}'.format(samples_left, samples_right))
    print('Values in left {}\tValues in right {}'.format(values_left, values_right))
    print('Class in left {}\t\tClass in right {}'.format(class_left, class_right))
    print('Entropy in left {:.3f}\tEntropy in right {:.3f}'.format(entropy_left, entropy_right))
    print('---' * 15)
    if entropy_left != 0:
        print('Splitting left leaf..')
        build_tree(X_left, y_left)
    else:
        print('Left leaf is optimal')
        print('---' * 15)
        print('\n')
        
    if entropy_right != 0:
        print('Splitting right leaf..')
        build_tree(X_right, y_right)
    else:
        print('Right leaf is optimal')
        print('---' * 15)
        print('\n')
# Toy data
X = pd.read_csv('data.csv', index_col=0)
y = pd.read_csv('labels.csv', index_col=0, squeeze=True)

# Test of the function
build_tree(X, y)