from abc import abstractmethod

class DatasetAligner:
    @abstractmethod
    def align_images(self, rgb_image, thermal_image):
        pass