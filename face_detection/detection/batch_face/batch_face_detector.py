from batch_face_detector import Retina_face
from ..core import FaceDetector
import torch
import os 

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mobilenet.pth')

class BFD(FaceDetector):
    def __init__(self, device, path_to_detector=model_path, verbose=False):
        super(BFD, self).__init__(device, verbose)
        self.face_detector = Retina_face(model_path)
        self.face_detector.load_state_dict(torch.load(model_path))
        self.face_detector.eval()

    def detect_from_image(self, tensor_or_path):
        image = self.tensor_or_path_to_ndarray(tensor_or_path)
        bboxlist = self.face_detector.detect(image)
        return bboxlist

    def detect_from_batch(self, images):
        bboxlists = self.face_detector.detect_batch(images)
        return bboxlists

    @property
    def reference_scale(self):
        return 195

    @property
    def reference_x_shift(self):
        return 0

    @property
    def reference_y_shift(self):
        return 0
