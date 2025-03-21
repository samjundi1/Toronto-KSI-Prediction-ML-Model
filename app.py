import flask
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_curve
from data_loder import DataLoader
from models import RandomForestEmander, HistGradientBoostingWonyoung, LogisticRegressionUtku, SupportVectorClassifierNilkanth
import os


app = flask.Flask(__name__)
app.secret_key = "COMP247"
plt.switch_backend("agg")
data_loader = DataLoader()
clfs = {}


@app.route("/train", methods=["POST"])
def train():
    global clfs
    data_loader.load_data()

    print("Preparing RF model")
    try:
        rf = joblib.load("rf.pkl")
    except:
        rf = RandomForestEmander()
        rf.train(data_loader.X_train_oversampled,
                 data_loader.y_train_oversampled)
        joblib.dump(rf, "rf.pkl")
    clfs["RandomForest-Emander"] = rf
    print("Done")

    print("Preparing HGB model")
    try:
        hgb = joblib.load("hgb.pkl")
    except:
        hgb = HistGradientBoostingWonyoung()
        hgb.train(data_loader.X_train_oversampled,
                  data_loader.y_train_oversampled)
        joblib.dump(hgb, "hgb.pkl")
    clfs["HistGradientBoosting-Wonyoung"] = hgb
    print("Done")

    print("Preparing LR model")
    try:
        lr = joblib.load("lr.pkl")
    except:
        lr = LogisticRegressionUtku()
        lr.train(data_loader.X_train_oversampled,
                 data_loader.y_train_oversampled)
        joblib.dump(lr, "lr.pkl")
    clfs["LogisticRegression-Utku"] = lr
    print("Done")

    print("Preparing SVM model")
    try:
        svm = joblib.load("svm.pkl")
    except:
        svm = SupportVectorClassifierNilkanth()
        svm.train(data_loader.X_train_oversampled,
                  data_loader.y_train_oversampled)
        joblib.dump(svm, "svm.pkl")
    clfs["SVM With Bagging-Nilkanth"] = svm
    print("Done")

    value_by_feature = data_loader.get_unique_values_by_features()
    dump = flask.json.dumps(value_by_feature)
    return dump, 200


@app.route("/predict", methods=["POST"])
def predict():
    json = flask.request.json
    x = pd.DataFrame([json["input_data"]])

    # Convert "NaN" to NaN
    x = x.replace("NaN", float("NaN"))

    x_processed = data_loader.preprocessor.transform(x)

    ret = {}
    for model_name, clf in clfs.items():
        pred = clf.predict(x_processed)[0]
        ret[model_name] = "Non-Fatal" if pred == 0 else "Fatal"

    return ret, 200


@app.route("/test", methods=["POST"])
def test():
    msg = ""
    X_test = data_loader.X_test
    y_test = data_loader.y_test

    for model_name, clf in clfs.items():
        msg += "Testing " + model_name + "\n"
        y_pred = clf.predict(X_test)

        msg += "Accuracy: {}\n".format(accuracy_score(y_test, y_pred))
        msg += "Precision: {}\n".format(precision_score(y_test,
                                        y_pred, average="weighted"))
        msg += "Recall: {}\n".format(recall_score(y_test,
                                     y_pred, average="weighted"))
        msg += "F1: {}\n".format(f1_score(y_test, y_pred, average="weighted"))
        msg += "Confusion matrix:\n{}\n\n".format(
            confusion_matrix(y_test, y_pred))

        # Save ROC curve
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        plt.plot([0, 1], [0, 1], "k--")
        plt.plot(fpr, tpr, label=model_name)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.savefig("ROC_" + model_name + ".png")
        plt.clf()

    return msg, 200


@app.route("/")
def index():
    return flask.render_template("index.html"), 200

# Run the Flask application
if __name__ == '__main__':
    #port = 5000
    port = int(os.environ.get("PORT", 5000))  # Render dynamically assigns a port
    app.run(host='0.0.0.0', debug=True)
    #pp.run(host='127.0.0.1', port=port, debug=True)