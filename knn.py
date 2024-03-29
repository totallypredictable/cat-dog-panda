from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imgp.preprocessing import SimplePreprocessor
from imgp.datasets import SimpleDatasetLoader
from imutils import paths
import argparse
import logging


def main():
    logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-d", "--dataset", required=True, help="path to input dataset"
    )
    ap.add_argument(
        "-k",
        "--neighbors",
        type=int,
        default=1,
        help="# of nearest neighbors for classification",
    )
    ap.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=-1,
        help="# of jobs for kNN distance (-1 uses all available cores)",
    )
    args = vars(ap.parse_args())

    logging.info("loading images...")
    imagePaths = list(paths.list_images(args["dataset"]))

    # initialize the image preprocessor, load the dataset from disk, and reshape
    # the data matrix
    sp = SimplePreprocessor(32, 32)
    sdl = SimpleDatasetLoader(preprocessors=[sp])
    (data, labels) = sdl.load(imagePaths, verbose=500)
    data = data.reshape((data.shape[0], 3072))

    logging.info(
        "features matrix: {:.1f}MB".format(data.nbytes / (1024 * 1000.0))
    )

    # encode the labels as integers
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    # for training and the remaining 25% for testing
    # partition the data into training and testing splits using 75% of the data
    (trainX, testX, trainY, testY) = train_test_split(
        data, labels, test_size=0.25, random_state=42
    )

    logging.info("evaluating kNN classifier...")
    model = KNeighborsClassifier(
        n_neighbors=args["neighbors"], n_jobs=args["jobs"]
    )
    model.fit(trainX, trainY)
    print(
        classification_report(
            testY, model.predict(testX), target_names=le.classes_
        )
    )


if __name__ == "__main__":
    main()
