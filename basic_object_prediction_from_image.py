import os

from imageai.Prediction import ImagePrediction


def predict_objects_densnet(filename):

    prediction = ImagePrediction()
    prediction.setModelTypeAsDenseNet()
    prediction.setModelPath(os.path.join(os.getcwd(), "models/DenseNet-BC-121-32.h5"))
    prediction.loadModel()

    print("----------DenseNet--------------")
    predictions, probabilities = prediction.predictImage(filename, result_count=20)
    for eachPrediction, eachProbability in zip(predictions, probabilities):
        print(eachPrediction, " : ", eachProbability)


def predict_objects_resnet(filename):

    prediction = ImagePrediction()
    prediction.setModelTypeAsResNet()
    prediction.setModelPath(os.path.join(os.getcwd(), "models/resnet50_weights_tf_dim_ordering_tf_kernels.h5"))
    prediction.loadModel()

    print("----------ResNet--------------")
    predictions, probabilities = prediction.predictImage(filename, result_count=20)
    for eachPrediction, eachProbability in zip(predictions, probabilities):
        print(eachPrediction, " : ", eachProbability)


if __name__ == "__main__":
    filename = "examples/pub.jpg"
    predict_objects_resnet(filename)
