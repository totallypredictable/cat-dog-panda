from imgp.nn import Perceptron
import logging
import numpy as np


def main():
    # construct the OR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [1]])

    # define our perceptron and train it
    logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

    logging.info("training perceptron...")
    p = Perceptron(X.shape[1], alpha=0.1)
    p.fit(X, y, epochs=20)

    logging.info("testing perceptron")
    for (x, target) in zip(X, y):
        # to make a prediction on the data point and display the result to
        # our console
        pred = p.predict(x)
        logging.info(
            "data={}, ground-trust={}, pred={}".format(x, target[0], pred)
        )


if __name__ == "__main__":
    main()
