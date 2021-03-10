# face_detetion_MTCNN
Detect face in image and video using mtcnn

```python
import cv2
import tensorflow
from mtcnn.mtcnn import MTCNN
detector = MTCNN()
image = cv2.imread(r'F:\\image\\my_pic\\1.jpg')
result = detector.detect_faces(image)


print(result)


'''[{'box': [91, 146, 56, 69],
  'confidence': 0.9916427135467529,
  'keypoints': {'left_eye': (101, 176),
   'right_eye': (126, 170),
   'nose': (112, 186),
   'mouth_left': (107, 200),
   'mouth_right': (131, 196)}}] '''

```
