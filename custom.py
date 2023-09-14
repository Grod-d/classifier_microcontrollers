import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2
from mrcnn.visualize import display_instances
import matplotlib.pyplot as plt

# выбор папки
ROOT_DIR = "D:\\projects\\microconrollers_rcnn"

# импорт Mask RCNN
sys.path.append(ROOT_DIR)  # моделька должна находиться в директории
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# название модельки
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# сохраняем логи при трейне
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


class CustomConfig(Config):
    """Конфигурация для обучения на пользовательском наборе данных..
    """
   
    NAME = "microcontrollers"

    
    IMAGES_PER_GPU = 2

    # Количество классов (включая фон)
    NUM_CLASSES = 1 + 5  

    
    STEPS_PER_EPOCH = 10

    # Пропускать обнаружения с вероятностью < 90%
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class CustomDataset(utils.Dataset):

    def load_custom(self, dataset_dir, subset):
        """Load a subset of the dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # нейминг классов
        self.add_class("microcontrollers", 1, "Arduino")
        self.add_class("microcontrollers", 2, "Adruino nano")
        self.add_class("microcontrollers", 3, "ESP8226")
        self.add_class("microcontrollers", 4, "Heltec")
        self.add_class("microcontrollers", 5, "Raspberry")

        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # В основном нас интересуют координаты x и y каждого региона
        annotations1 = json.load(open('..\dataset\train\_annotations.coco.json'))
        # print(annotations1)
        annotations = list(annotations1.values()) 

        # VIA сохраняет изображения в формате JSON, даже если к ним нет никаких аннотаций. пропускаем неаннотированные изображения.
        annotations = [a for a in annotations if a['regions']]

        for a in annotations:
            # print(a)
            # координаты x, y точек многоугольников, составляющих контур каждого экземпляра объекта
            polygons = [r['shape_attributes'] for r in a['regions']]
            objects = [s['region_attributes']['names'] for s in a['regions']]
            print("objects:", objects)
            name_dict = {"Arduino": 1, "Adruino nano": 2, "ESP8226": 3, "Heltec": 4, "Raspberry": 5}

            # key = tuple(name_dict)
            num_ids = [name_dict[a] for a in objects]

            # num_ids = [int(n['Event']) for n in objects]
            # load_mask() необходим размер изображения для преобразования полигонов в маски.
            # VIA не включает его в JSON, поэтому мы должны прочитать изображение.
            # Это выполнимо только потому, что набор данных небольшой.
            print("numids", num_ids)
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "object",  ## 
                image_id=a['filename'],  #
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                num_ids=num_ids
            )

    def load_mask(self, image_id):
        """Генерация маски экземпляров для изображения.
        Возвращается:
        маски: массив bool [высота, ширина, количество экземпляров]
        одна маска на экземпляр.
        class_ids: одномерный массив идентификаторов классов масок экземпляров.
        """

        image_info = self.image_info[image_id]
        if image_info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)

        # Преобразование полигонов в растровую маску формы [высота, ширина, instance_count]
        info = self.image_info[image_id]
        if info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)
        num_ids = info['num_ids']
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            #Получаем индексы пикселей внутри многоугольника и устанавливаем их равными 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])

            mask[rr, cc, i] = 1

        # Возвращаемая маска и массив идентификаторов классов каждого экземпляра. Поскольку у нас есть
        # только один идентификатор класса, мы возвращаем массив из 1s
        # Сопоставляем имена классов с идентификаторами классов.
        num_ids = np.array(num_ids, dtype=np.int32)
        return mask, num_ids  # np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "object":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = CustomDataset()
    dataset_train.load_custom("..\dataset", "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CustomDataset()
    dataset_val.load_custom("..\dataset", "valid")
    dataset_val.prepare()

    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=20,
                layers='heads')


config = CustomConfig()
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=DEFAULT_LOGS_DIR)

weights_path = COCO_WEIGHTS_PATH

if not os.path.exists(weights_path):
    utils.download_trained_weights(weights_path)

model.load_weights(weights_path, by_name=True, exclude=[
    "mrcnn_class_logits", "mrcnn_bbox_fc",
    "mrcnn_bbox", "mrcnn_mask"])

train(model)
