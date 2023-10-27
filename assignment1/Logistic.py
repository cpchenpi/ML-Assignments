import numpy as np


class LogisticRegression:
    def __init__(self, penalty="l2", gamma=0.0, fit_intercept=True):
        """
        Parameters:
        - penalty: str, "l1" or "l2". Determines the regularization to be used.
        - gamma: float, regularization coefficient. Used in conjunction with 'penalty'.
        - fit_intercept: bool, whether to add an intercept (bias) term.
        """
        err_msg = "penalty must be 'l1' or 'l2', but got: {}".format(penalty)
        assert penalty in ["l2", "l1"], err_msg
        self.penalty = penalty
        self.gamma = gamma
        self.fit_intercept = fit_intercept
        self.coef = np.array(0)

    def sigmoid(self, i):
        """The logistic sigmoid function"""
        ################################################################################
        # TODO:                                                                        #
        # Implement the sigmoid function.
        ################################################################################
        if i >= 0:
            return 1.0 / (1 + np.exp(-i))
        else:
            return np.exp(i) / (1.0 + np.exp(i))

        ################################################################################
        #                                 END OF YOUR CODE                             #
        ################################################################################

    def loss(self, X, Y):
        res = 0.0
        for x, y in zip(X, Y):
            i = np.dot(x, self.coef)
            if i >= 0:
                res += -y * i + i + np.log(1 + np.exp(-i))
            else:
                res += -y * i + np.log(1 + np.exp(i))
        res /= X.shape[0]

        if self.penalty == "l1":
            res += self.gamma * np.linalg.norm(self.coef, ord=1)
        if self.penalty == "l2":
            res += self.gamma * np.dot(self.coef, self.coef)
        return res

    def gradient(self, X, Y, eps=1e-7):
        res = np.zeros(self.coef.shape)
        for x, y in zip(X, Y):
            res += x * (self.sigmoid(np.dot(x, self.coef)) - y)
        res /= X.shape[0]

        if self.penalty == "l1":
            for i, w in enumerate(self.coef):
                if w > eps:
                    res[i] += self.gamma
                elif w < -eps:
                    res[i] -= self.gamma
        if self.penalty == "l2":
            res += 2 * self.gamma * self.coef
        return res

    def fit(self, X, y, lr=0.01, tol=1e-7, max_iter=10**7, test_X=None, test_y=None):
        """
        Fit the regression coefficients via gradient descent or other methods

        Parameters:
        - X: numpy array of shape (n_samples, n_features), input data.
        - y: numpy array of shape (n_samples,), target data.
        - lr: float, learning rate for gradient descent.
        - tol: float, tolerance to decide convergence of gradient descent.
        - max_iter: int, maximum number of iterations for gradient descent.
        Returns:
        - losses: list, a list of loss values at each iteration.
        """
        # If fit_intercept is True, add an intercept column
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        # Initialize coefficients
        self.coef = np.zeros(X.shape[1])

        # List to store loss values at each iteration
        losses = []
        losses.append(self.loss(X, y))
        acc = []
        if test_X is not None:
            acc.append(self.predict_acc(test_X, test_y))

        ################################################################################
        # TODO:                                                                        #
        # Implement gradient descent with optional regularization.
        # 1. Compute the gradient
        # 2. Apply the update rule
        # 3. Check for convergence
        ################################################################################

        for epoch in range(0, max_iter):
            self.coef -= lr * self.gradient(X, y)
            losses.append(self.loss(X, y))
            if test_X is not None:
                acc.append(self.predict_acc(test_X, test_y))
            if abs(losses[-1] - losses[-2]) < tol:
                break

        ################################################################################
        #                                 END OF YOUR CODE                             #
        ################################################################################
        return losses, acc

    def predict(self, X):
        """
        Use the trained model to generate prediction probabilities on a new
        collection of data points.

        Parameters:
        - X: numpy array of shape (n_samples, n_features), input data.

        Returns:
        - probs: numpy array of shape (n_samples,), prediction probabilities.
        """
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        # Compute the linear combination of inputs and weights
        linear_output = np.dot(X, self.coef)

        ################################################################################
        # TODO:                                                                        #
        # Task3: Apply the sigmoid function to compute prediction probabilities.
        ################################################################################

        return [self.sigmoid(i) for i in linear_output]

        ################################################################################
        #                                 END OF YOUR CODE                             #
        ################################################################################

    def predict_acc(self, X, Y):
        p = self.predict(X)
        c = 0
        n = Y.shape[0]

        for y, p in zip(Y, p):
            p = int(p >= 0.5)
            c += int(y == p)
        return c / n
