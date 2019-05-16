import os
import dill

import numpy as np
import data_pre_processing as dp


from keras import backend as k
from keras import optimizers
from keras.callbacks import Callback
from keras.layers import Input, Dense, concatenate, add
from keras.models import Model, load_model
from keras.regularizers import l1
from itertools import chain
from sklearn.metrics import accuracy_score
from shutil import copyfile


class StopEarly(Callback):
    def __init__(self, threshold, metric="val_acc", verbose=True):
        super(StopEarly, self).__init__()
        self.threshold = threshold
        self.metric = metric
        self.last_value = 0
        self.stopped_epoch = 0
        self.notChanged = 0
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.metric)
        if logs.get(self.metric) - self.last_value < self.threshold:
            if self.notChanged >= 3:
                self.model.stop_training = True
                self.stopped_epoch = epoch
            else:
                self.notChanged += 1
        else:
            self.notChanged = 0
        self.last_value = current

    def on_train_end(self, log={}):
        if self.stopped_epoch > 0 and self.verbose:
            print(
                "model stopped training on epoch",
                self.stopped_epoch,
                "with val_acc =",
                self.last_value,
            )


def runthrough(T, depth, layerDic):
    for i in range(depth):
        for t in range(T):
            for prefix in ("c", ""):
                name = prefix + str(i) + "." + str(t)
                try:
                    print(name, layerDic[name])
                except:
                    pass
    for name in ("c.out", "output.Layer"):
        try:
            print(name, layerDic[name])
        except:
            pass
    print("\n\n")


def toSymbolicDict(T, depth, layerDic):
    tensorDic = {}
    key = "feeding.Layer"
    params = layerDic[key]
    tensorDic[key] = Input(shape=params[1]["shape"], name=key)

    for i in range(depth):
        for t in range(T):
            for prefix in ("c", ""):
                key = prefix + str(i) + "." + str(t)
                try:
                    params = layerDic[key]
                    if key[0] == "c":  # cocatenating layer
                        candidateLayers = Call(tensorDic, params[1])
                        tensorDic[key] = params[0](candidateLayers)
                    elif key != "output.Layer":
                        tensorDic[key] = params[0](
                            params[1]["units"],
                            activation=params[1]["activation"],
                            name=key,
                        )(tensorDic[params[2]])
                except:
                    pass

    key = "c.out"
    params = layerDic[key]
    candidateLayers = Call(tensorDic, params[1])
    tensorDic[key] = params[0](candidateLayers)
    key = "output.Layer"
    params = layerDic[key]
    tensorDic[key] = params[0](
        params[1]["units"], activation=params[1]["activation"], name=key
    )(tensorDic[params[2]])

    return tensorDic


def build_new(
    B,
    T,
    flat_image,
    lr,
    reps,
    x_train,
    y_train,
    x_test,
    y_test,
    epochs,
    size,
    epsilon,
    pathToSaveModel,
    probaThreshold,
    handleMultipleInput,
    lambda1,
):
    count = 1
    layerDic = {}
    layersNamesToOutput = []
    concatOutName = "c.out"

    earlyStopping = StopEarly(0.0001, "val_acc", True)

    layerDic["feeding.Layer"] = (
        Input,
        {"shape": (flat_image,), "name": "feeding.Layer"},
    )

    for t in range(T):
        changed = False  # boolean to track if the base model has improved (improved)
        print("\n\n" + 100 * "=" + "\niteration n." + str(t) + "\n" + 100 * "=")
        if t == 0:
            layerName = "0.0"
            layerDic[layerName] = (
                Dense,
                {
                    "units": B,
                    "activation": "relu",
                    "kernel_regularizer": l1(lambda1),
                    "name": layerName,
                },
                "feeding.Layer",
            )
            layerDic["output.Layer"] = (
                Dense,
                {
                    "units": 1,
                    "activation": "sigmoid",
                    "kernel_regularizer": l1(lambda1),
                    "name": "output.Layer",
                },
                layerName,
            )
            layersNamesToOutput.append(layerName)
            previousScore = float("Inf")

            symbolicTensorsDict = toSymbolicDict(1, 1, layerDic)
            model = Model(
                inputs=symbolicTensorsDict["feeding.Layer"],
                outputs=symbolicTensorsDict["output.Layer"],
            )
            model.compile(
                optimizer=optimizers.SGD(lr=lr, decay=1e-6, momentum=0.5),
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )
            model.fit(
                x=x_train,
                y=y_train,
                validation_split=0.1,
                callbacks=[earlyStopping],
                epochs=epochs,
                batch_size=size,
                verbose=0,
            )
            model.save_weights("w_" + pathToSaveModel)
            model.save(pathToSaveModel)

            with open("layerDic.pkl", "wb") as dicFile:
                dill.dump(layerDic, dicFile)
            with open("layersNamesToOutput.pkl", "wb") as outFile:
                dill.dump(layersNamesToOutput, outFile)
            k.clear_session()
        else:

            if t > 1:
                copyfile(pathToSaveModel, str(t - 1) + pathToSaveModel)
                copyfile("w_" + pathToSaveModel, "w_" + str(t - 1) + pathToSaveModel)
                try:
                    os.rename("best_" + pathToSaveModel, pathToSaveModel)
                    os.rename("best_w_" + pathToSaveModel, "w_" + pathToSaveModel)
                except:
                    pass

            for rep in range(reps):
                print("\n rep " + str(rep))

                modelTest = load_model(pathToSaveModel)

                previousDepth = getPreviousDepth(layerDic, t)
                previousPredictions = classPrediction(modelTest, x_test, probaThreshold)

                with open("layerDic.pkl", "rb") as f:
                    layerDic = dill.load(f)
                with open("layersNamesToOutput.pkl", "rb") as f:
                    layersNamesToOutput = dill.load(f)
                currentDepth = previousDepth + 1

                for depth in range(currentDepth):
                    layerName = str(depth) + "." + str(t)

                    concatLayerName = "c" + layerName
                    if handleMultipleInput == "concatenate":
                        functionChoice = concatenate
                    elif handleMultipleInput == "add":
                        functionChoice = add
                    else:
                        raise ValueError(
                            "handleMultipleInput must have a value in ('concatenate','add')"
                        )

                    if depth == 0:
                        layerDic[layerName] = (
                            Dense,
                            {"units": B, "activation": "relu", "name": layerName},
                            "feeding.Layer",
                        )
                    else:
                        candidateNameList = selectCandidateLayers(layerDic, t, depth)
                        candidateNameList = drawing(candidateNameList)
                        layerBelowName = str(depth - 1) + "." + str(t)
                        candidateNameList.append(layerBelowName)
                        candidateNameList = list(set(candidateNameList))
                        if len(candidateNameList) > 1:
                            layerDic[concatLayerName] = (
                                functionChoice,
                                candidateNameList,
                            )
                            layerDic[layerName] = (
                                Dense,
                                {
                                    "units": B,
                                    "activation": "relu",
                                    "kernel_regularizer": l1(lambda1),
                                    "name": layerName,
                                },
                                concatLayerName,
                            )
                        else:
                            layerDic[layerName] = (
                                Dense,
                                {
                                    "units": B,
                                    "activation": "relu",
                                    "kernel_regularizer": l1(lambda1),
                                    "name": layerName,
                                },
                                candidateNameList[0],
                            )
                    if depth == currentDepth - 1:
                        layersNamesToOutput.append(layerName)

                if len(layersNamesToOutput) > 1:
                    layerDic[concatOutName] = (
                        functionChoice,
                        list(set(layersNamesToOutput)),
                    )
                    layerDic["output.Layer"] = (
                        Dense,
                        {
                            "units": 1,
                            "activation": "sigmoid",
                            "kernel_regularizer": l1(lambda1),
                            "name": "output.Layer",
                        },
                        concatOutName,
                    )
                else:
                    layerDic["output.Layer"] = (
                        Dense,
                        {
                            "units": 1,
                            "activation": "sigmoid",
                            "kernel_regularizer": l1(lambda1),
                            "name": "output.Layer",
                        },
                        layersNamesToOutput[0],
                    )

                symbolicTensorsDict = toSymbolicDict(t + 1, currentDepth + 1, layerDic)

                model = Model(
                    inputs=symbolicTensorsDict["feeding.Layer"],
                    outputs=symbolicTensorsDict["output.Layer"],
                )
                model.compile(
                    optimizer=optimizers.SGD(
                        lr=lr, momentum=0.8
                    ),
                    loss="binary_crossentropy",
                    metrics=["accuracy"],
                )

                model.load_weights("w_" + pathToSaveModel, by_name=True)

                model.fit(
                    x=x_train,
                    y=y_train,
                    validation_split=0.1,
                    callbacks=[earlyStopping],
                    epochs=epochs,
                    batch_size=size,
                    verbose=1,
                )
                print("fitted model number ", count)
                count += 1
                currentPredictions = classPrediction(model, x_test, probaThreshold)
                currentScore = objectiveFunction(
                    y_test, previousPredictions, currentPredictions
                )

                if previousScore - currentScore > epsilon:
                    print("saving better model")
                    changed = True

                    previousScore = currentScore
                    model.save("best_" + pathToSaveModel)
                    model.save_weights("best_w_" + pathToSaveModel)
                    with open("layersNamesToOutput.pkl", "wb") as f:
                        dill.dump(layersNamesToOutput, f)
                    with open("layerDic.pkl", "wb") as f:
                        dill.dump(layerDic, f)
                k.clear_session()
            if not changed:
                print("model not improved at iteration", t, "stopping early")
                return
    bestModel = load_model("best_" + pathToSaveModel)
    print(bestModel.metric_names)
    print("Test metrics : ", bestModel.evaluate(x_test, y_test))
    k.clear_session()

def drawing(candidates):
    result = np.random.choice(candidates, size=np.random.randint(0, len(candidates)), replace=False)
    return result.tolist()


def getPreviousDepth(layerDic, t):
    previousDepth = 0
    for layerName in layerDic.keys():
        depth, iteration = layerName.split(".")
        try:
            depth_int, iteration_int = int(depth), int(iteration)
            if iteration_int == t - 1 and depth_int > previousDepth:
                previousDepth = depth_int
        except:
            pass
    return previousDepth + 1


def selectCandidateLayers(layerDic, t, c):

    candidateList = []
    for layerName in layerDic.keys():
        depth, iteration = layerName.split(".")
        try:
            depth_int, iteration_int = int(depth), int(iteration)
            if depth_int == c - 1:
                candidateList.append(layerName)
        except:
            pass
    return candidateList


def Call(dic, keys):
    return [dic[key] for key in keys]


def classPrediction(model, x, probaThreshold):
    probas = np.array(model.predict(x))
    booleans = probas >= probaThreshold
    booleans = list(chain(*booleans))
    classes = []
    for boolean in booleans:
        if boolean:
            classes.append(1)
        else:
            classes.append(-1)
    return classes


def objectiveFunction(trueLabels, previousPredictions, currentPredictions):

    result = 0
    for i in range(len(trueLabels)):
        result += np.exp(
            1
            - trueLabels[i] * previousPredictions[i]
            - trueLabels[i] * currentPredictions[i]
        )
    result = result / len(trueLabels)
    return result


def Encoding(y_vect):
    """
	encodes two classes as labels 0 and 1
	"""
    return np.array([0 if i == y_vect[0] else 1 for i in y_vect])


def main():
    pathToSaveModel = "bestModel.h5"
    imsize = 32
    flattenDimIm = imsize * imsize * 3
    B = 150
    T = 10
    lr = 10e-3
    reps = 5
    trainNum = 5000
    testNum = 1000
    epochs = 1000
    batchSize = 32
    epsilon = 0.0001
    labels = [3, 5]
    probaThreshold = 0.5
    handleMultipleInput = "add"
    lambda1 = 0.000001

    print("B", B)
    print("T", T)
    print("lr", lr)
    print("epsilon", epsilon)
    print("labels", labels)
    print("lambda1", lambda1)

    if len(labels) > 2 or labels[0] == labels[1]:
        raise ValueError("labels must be array of 2 distinct values")
    for i in range(2):
        if labels[i] < 0 or labels[i] > 9:
            raise ValueError("label value must be between 0 and 9 included")

    train, test = dp.loadRawData()
    x_train, y_train = dp.loadTrainingData(train, labels, trainNum)
    x_test, y_test = dp.loadTestingData(test, labels, testNum)

    x_train = x_train.flatten().reshape(trainNum, flattenDimIm) / 255
    x_test = x_test.flatten().reshape(testNum, flattenDimIm) / 255

    y_train = Encoding(y_train)
    y_test = Encoding(y_test)

    build_new(
        B,
        T,
        flattenDimIm,
        lr,
        reps,
        x_train,
        y_train[:trainNum],
        x_test,
        y_test,
        epochs,
        batchSize,
        epsilon,
        pathToSaveModel,
        probaThreshold,
        handleMultipleInput,
        lambda1,
    )

    model = load_model(pathToSaveModel)

    preds = np.round(model.predict(x_test))
    print(accuracy_score(preds, y_test))


if __name__ == "__main__":
    main()
