import numpy as np
import test as T
import sys

FILE_PATH = "/Users/nyismaw/Documents/sample_submission_file"
model_predictions = ["/Users/nyismaw/Documents/sample_submission_file4.csv",
                    "/Users/nyismaw/Documents/sample_submission_file5.csv",
                    "/Users/nyismaw/Documents/sample_submission_file6.csv"]

                    
scores = np.zeros((3774, 5))


def return_class_index(label):
    if label == "cbb":
        return 0
    if label == "cbsd":
        return 1
    if label == "cgm":
        return 2
    if label == "cmd":
        return 3
    return 4


def get_class_from_id(idx):
    classes = []
    for id in idx:
        if id == 0:
            classes.append("cbb")
        elif id == 1:
            classes.append("cbsd")
        elif id == 2:
            classes.append("cgm")
        elif id == 3:
            classes.append("cmd")
        elif id == 4:
            classes.append("healthy")
    return classes


for model_prediction in model_predictions:
    data = np.loadtxt(model_prediction, dtype='str', delimiter=",")
    print(data.shape)
    for i in range(1, data.shape[0]):
        scores[i - 1, return_class_index(data[i, 0])] += 1


def main(argv):

    max_scores = np.argmax(scores, axis=1)
    print("Max scores shape is ", max_scores.shape)
    pred_labels = get_class_from_id(max_scores)
    sample_data = np.loadtxt(FILE_PATH + str(4) + ".csv",
                             dtype='str', delimiter=",")
    sample_data[1:, 0] = pred_labels
    np.savetxt("sub_ensamble.csv", sample_data, fmt="%s",  delimiter=',')


if __name__ == '__main__':
    main(sys.argv)
