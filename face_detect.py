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





 # image having multiple faces 
image = cv2.imread(r'F:\\image\\my_pic\\test2.jpg')
result = detector.detect_faces(image)

print(result)


'''
[{'box': [372, 156, 31, 41],
  'confidence': 0.9993925094604492,
  'keypoints': {'left_eye': (380, 173),
   'right_eye': (394, 171),
   'nose': (387, 181),
   'mouth_left': (382, 188),
   'mouth_right': (396, 186)}},
 {'box': [226, 40, 30, 39],
  'confidence': 0.9991531372070312,
  'keypoints': {'left_eye': (233, 56),
   'right_eye': (248, 54),
   'nose': (241, 63),
   'mouth_left': (234, 70),
   'mouth_right': (250, 69)}},
 {'box': [432, 44, 31, 43],
  'confidence': 0.9988896250724792,
  'keypoints': {'left_eye': (439, 63),
   'right_eye': (454, 60),
   'nose': (448, 70),
   'mouth_left': (443, 78),
   'mouth_right': (457, 75)}},
 {'box': [216, 263, 33, 43],
  'confidence': 0.9987927675247192,
  'keypoints': {'left_eye': (223, 280),
   'right_eye': (238, 279),
   'nose': (228, 290),
   'mouth_left': (224, 295),
   'mouth_right': (239, 295)}},
 {'box': [321, 269, 32, 40],
  'confidence': 0.998678982257843,
  'keypoints': {'left_eye': (330, 286),
   'right_eye': (346, 285),
   'nose': (339, 296),
   'mouth_left': (331, 301),
   'mouth_right': (345, 301)}},
 {'box': [133, 30, 28, 38],
  'confidence': 0.9975391626358032,
  'keypoints': {'left_eye': (141, 45),
   'right_eye': (156, 45),
   'nose': (149, 53),
   'mouth_left': (141, 59),
   'mouth_right': (155, 59)}},
 {'box': [160, 149, 30, 39],
  'confidence': 0.9971550703048706,
  'keypoints': {'left_eye': (168, 165),
   'right_eye': (183, 163),
   'nose': (176, 173),
   'mouth_left': (169, 179),
   'mouth_right': (184, 178)}},
 {'box': [259, 145, 32, 42],
  'confidence': 0.9939432740211487,
  'keypoints': {'left_eye': (269, 162),
   'right_eye': (284, 163),
   'nose': (276, 173),
   'mouth_left': (268, 177),
   'mouth_right': (283, 177)}},
 {'box': [62, 149, 32, 39],
  'confidence': 0.9919396042823792,
  'keypoints': {'left_eye': (74, 164),
   'right_eye': (89, 164),
   'nose': (83, 172),
   'mouth_left': (74, 178),
   'mouth_right': (88, 178)}},
 {'box': [506, 154, 30, 41],
  'confidence': 0.9856187701225281,
  'keypoints': {'left_eye': (518, 168),
   'right_eye': (532, 171),
   'nose': (526, 179),
   'mouth_left': (516, 183),
   'mouth_right': (529, 186)}},
 {'box': [452, 260, 32, 42],
  'confidence': 0.9774866104125977,
  'keypoints': {'left_eye': (464, 277),
   'right_eye': (479, 277),
   'nose': (474, 285),
   'mouth_left': (464, 293),
   'mouth_right': (476, 294)}},
 {'box': [104, 267, 31, 45],
  'confidence': 0.9743489027023315,
  'keypoints': {'left_eye': (110, 286),
   'right_eye': (126, 284),
   'nose': (117, 295),
   'mouth_left': (112, 300),
   'mouth_right': (128, 299)}}]
   '''
   

   