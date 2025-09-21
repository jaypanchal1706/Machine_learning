import numpy as np 

class NaiveByes:

    def fit(self, X, y):
        # fit the models to the training data X and y(labels)
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)  

        # calculate mean, var, prior for each class
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)        

        # Loop through each class and calculate stats 
        for idx, c in enumerate(self._classes):
            X_c = X[y==c]   # select all the samples in class c 
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0] / float(n_samples) 
    
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)     
    
    def _predict(self, x):
        posteriors = []

        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            class_conditional = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)
        
        return self._classes[np.argmax(posteriors)]
    
    def _pdf(self, class_idx, x):       
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(- (x - mean)**2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator  
    
# Testing 

if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Load the dataset 
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    # Split the dataset into training and test set 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    # Train the model 
    clf = NaiveByes()
    clf.fit(X_train, y_train)

    # Make predictions 
    predictions = clf.predict(X_test)

    # Calculate the accuracy 
    acc = accuracy_score(y_test, predictions)
    print("Accuracy:", acc) 
