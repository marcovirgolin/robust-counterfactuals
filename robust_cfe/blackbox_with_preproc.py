from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class BlackboxWithPreproc(BaseEstimator,ClassifierMixin):
    
    def __init__(self, blackbox, indices_categorical_features=list(), preprocs=['onehot']):
        self.blackbox = blackbox
        self.indices_categorical_features = indices_categorical_features if indices_categorical_features is not None else list()
        self.preprocs = preprocs

    def __apply_preproc(self, X, fit=False):
        X_prime = X.copy()

        if self.preprocs is None:
            return X_prime

        for preproc in self.preprocs:
            if preproc == 'onehot':
                # identify categorical columns with more than two categories
                if fit:
                    self.nonbinary_cat_features = list()
                    for d in self.indices_categorical_features:
                        if len(np.unique(X_prime[:,d])) > 2:
                            self.nonbinary_cat_features.append(d)
                if len(self.nonbinary_cat_features) == 0:
                    continue
                X_cat = X_prime[:,self.nonbinary_cat_features]
                if fit:
                    self.ohe = OneHotEncoder(handle_unknown='ignore')
                    X_ohe = self.ohe.fit_transform(X_cat).toarray()
                else:
                    X_ohe = self.ohe.transform(X_cat).toarray()
                X_prime = np.delete(X_prime, self.nonbinary_cat_features, axis=1)
                X_prime = np.concatenate((X_prime, X_ohe), axis=1)

            elif preproc == 'standard_scale' or preproc == 'minmax_scale':
                num_feature_indices = [d for d in X.shape[1] if d not in self.indices_categorical_features]
                X_num = X_prime[:, num_feature_indices]
                if fit:
                    if preproc == 'standard_scale':
                        self.scaler = StandardScaler()
                    elif preproc == 'minmax_scale':
                        self.scaler = MinMaxScaler()
                    X_scaled = self.scaler.fit_transform(X_num)
                else:
                    X_scaled = self.scaler.transform(X_num)
                X_prime = np.delete(X_prime, num_feature_indices, axis=1)
                X_prime = np.concatenate((X_prime, X_scaled), axis=1)
            else:
                raise ValueError("Unknown preprocessing:", preproc)

        return X_prime


    def fit(self, X, y):
        X_prime = self.__apply_preproc(X, fit=True)
        self.blackbox.fit(X_prime, y)

    def predict(self, X):
        if type(X) is int or type(X) is float:
            X = np.array([X])
        elif type(X) != np.ndarray:
            X = np.array(X)

        X_prime = self.__apply_preproc(X, fit=False)
        pred = self.blackbox.predict(X_prime)
        return pred

    def predict_proba(self, X):
        if type(X) is int or type(X) is float:
            X = np.array([X])
        elif type(X) != np.ndarray:
            X = np.array(X)

        X_prime = self.__apply_preproc(X, fit=False)
        proba = self.blackbox.predict_proba(X_prime)
        return proba


    def score(self, X, y=None):
        if y is None:
            raise ValueError("The ground truth y was not set")

        if type(X) is int or type(X) is float:
            X = np.array([X])
        elif type(X) != np.ndarray:
            X = np.array(X)

        X_prime = self.__apply_preproc(X, fit=False)
        score = self.blackbox.score(X_prime, y)
        return score
        