import numpy as np


class CustomSVC:
    def __init__(self, step_size=0.001, reg_param=0.01, iterations=1000):
        self.step = step_size
        self.reg = reg_param
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def fit(self, features, targets):
        num_samples, num_features = features.shape

        # Convert labels: values <= 0 become -1, others become 1.
        transformed_targets = np.where(targets <= 0, -1, 1)

        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.iterations):
            for idx, sample in enumerate(features):
                cond = transformed_targets[idx] * (np.dot(sample, self.weights) - self.bias) >= 1
                if cond:
                    self.weights -= self.step * (2 * self.reg * self.weights)
                else:
                    self.weights -= self.step * (
                        2 * self.reg * self.weights - np.dot(sample, transformed_targets[idx])
                    )
                    self.bias -= self.step * transformed_targets[idx]

    def predict(self, features):
        approximation = np.dot(features, self.weights) - self.bias
        return np.sign(approximation)


# Testing the classifier
if __name__ == "__main__":
    from sklearn import datasets
    import matplotlib.pyplot as plt

    # Generate sample data
    data, labels = datasets.make_blobs(
        n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40
    )
    labels = np.where(labels == 0, -1, 1)

    model = CustomSVC()
    model.fit(data, labels)
    print(model.weights, model.bias)

    def plot_decision_boundary():
        def calc_hyperplane(x_val, weights, bias, offset):
            return (-weights[0] * x_val + bias + offset) / weights[1]

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.scatter(data[:, 0], data[:, 1], marker="o", c=labels)

        x_min = np.amin(data[:, 0])
        x_max = np.amax(data[:, 0])

        y_line_0_min = calc_hyperplane(x_min, model.weights, model.bias, 0)
        y_line_0_max = calc_hyperplane(x_max, model.weights, model.bias, 0)

        y_line_neg_min = calc_hyperplane(x_min, model.weights, model.bias, -1)
        y_line_neg_max = calc_hyperplane(x_max, model.weights, model.bias, -1)

        y_line_pos_min = calc_hyperplane(x_min, model.weights, model.bias, 1)
        y_line_pos_max = calc_hyperplane(x_max, model.weights, model.bias, 1)

        ax.plot([x_min, x_max], [y_line_0_min, y_line_0_max], "y--")
        ax.plot([x_min, x_max], [y_line_neg_min, y_line_neg_max], "k")
        ax.plot([x_min, x_max], [y_line_pos_min, y_line_pos_max], "k")

        y_min = np.amin(data[:, 1])
        y_max = np.amax(data[:, 1])
        ax.set_ylim([y_min - 3, y_max + 3])

        plt.show()

    plot_decision_boundary()