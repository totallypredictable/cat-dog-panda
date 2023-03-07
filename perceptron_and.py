from imgp.nn import Perceptron
import logging
import numpy as np


def main():
    logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

    # construct the AND dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [0], [0], [1]])

    # define our perceptron and train it
    logging.info("training perceptron...")
    p = Perceptron(X.shape[1], alpha=0.1)
    p.fit(X, y, epochs=20)

    # now that our perceptron is trained, we can evaluate iter
    logging.info("testing perceptron...")

    # loop over the data points
    for (x, target) in zip(X, y):
        # make a prediction on the data point and display the result to console
        pred = p.predict(x)
        logging.info(
            "data={}, ground-truth={}, pred={}".format(x, target[0], pred)
        )


if __name__ == "__main__":
    main()
