import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report
class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate  
        self.lambda_param = lambda_param  
        self.n_iters = n_iters  
        self.w = 0  
        self.b = 0 

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1) 

        self.w = np.zeros(n_features)  

        for _ in range(self.n_iters):  
            for idx, x in enumerate(X):
                condition = y_[idx] * (np.dot(x, self.w) + self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        linear_output = np.dot(X, self.w) + self.b
        return np.sign(linear_output)  

if __name__ == "__main__":

    X , y =load_breast_cancer(return_X_y=True)

    svm = SVM(learning_rate=0.01, lambda_param=0.1, n_iters=1000)
    svm.fit(X, y)

    predictions = svm.predict(X)
    actual_pred = np.where(predictions <= 0, 0, 1) 
    print("Predictions:", actual_pred)
    print(classification_report(y , actual_pred))