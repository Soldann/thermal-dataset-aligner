from abc import abstractmethod

class DatasetAligner:
    def __init__(self):
        self.debug_mode = False

    @abstractmethod
    def align_images(self, rgb_image, thermal_image):
        pass