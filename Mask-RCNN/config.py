from mrcnn.config import Config

class CellsConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "cells"

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 cell
    IMAGE_MIN_DIM = 900
    STEPS_PER_EPOCH = 10000



    IMAGES_PER_GPU = 1
   



config = CellsConfig()
