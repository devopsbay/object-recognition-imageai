from imageai.Detection.Custom import DetectionModelTrainer
from imageai.Prediction.Custom import ModelTraining


def train_model():
    trainer = DetectionModelTrainer()
    trainer.setModelTypeAsYOLOv3()
    trainer.setDataDirectory(data_directory="examples/specjale")
    trainer.setTrainConfig(object_names_array=["specjal"], batch_size=4, num_experiments=10,
                           train_from_pretrained_model="models/yolo.h5")
    trainer.trainModel()


if __name__ == "__main__":
    train_model()
