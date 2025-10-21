from abc import abstractmethod

class FeatureMatcher:
    def __init__(self):
        self.debug_mode = False

    @abstractmethod
    def compute_correspondences(self, image1, image2):
        pass