from batch_face import RetinaFace
from ..core import FaceDetector
import torch
import os 
import math

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mobilenet.pth')

class BFD(FaceDetector):
    def __init__(self, device, path_to_detector=model_path, verbose=False):
        super(BFD, self).__init__(device, verbose)
        self.face_batch_size = 64 * 1 
        self.face_detector = RetinaFace(gpu_id=0, model_path=model_path, network='mobilenet')
        
    def detect_from_image(self, tensor_or_path):
        image = self.tensor_or_path_to_ndarray(tensor_or_path)
        bboxlist = self.face_detector.detect(image)
        return bboxlist

    def detect_from_batch(self, images):
        num_batches = math.ceil(len(images) / self.face_batch_size)
        for i in range(num_batches):
            batch = images[i * self.face_batch_size: (i + 1) * self.face_batch_size]
            all_faces = self.face_rect(batch)  # Your batch detection implementation

            batch_rects = []
            for faces in all_faces:
                if faces:
                    for face in faces:
                        box, landmarks, score = face
                        box = tuple(map(int, box))
                        batch_rects.append(box)
            yield batch_rects

    
    def face_rect(self, images):
        num_batches = math.ceil(len(images) / self.face_batch_size)
        prev_ret = None
        for i in range(num_batches):
            batch = images[i * self.face_batch_size: (i + 1) * self.face_batch_size]
            all_faces = self.face_detector(batch)  # return faces list of all images
            for faces in all_faces:
                if faces:
                    box, landmarks, score = faces[0]
                    prev_ret = tuple(map(int, box))
                yield prev_ret

    @property
    def reference_scale(self):
        return 195

    @property
    def reference_x_shift(self):
        return 0

    @property
    def reference_y_shift(self):
        return 0
