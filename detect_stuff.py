import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

FROZEN_GRAPH_FILE = 'frozen_inference_graph.pb'

graph = tf.Graph()
with graph.as_default():
    serialGraph = tf.GraphDef()
    with tf.gfile.GFile(FROZEN_GRAPH_FILE, 'rb') as f:
        serialRead = f.read()
        serialGraph.ParseFromString(serialRead)
        tf.import_graph_def(serialGraph, name='')

sess = tf.Session(graph=graph)

image = cv2.imread('test.jpg')

imageTensor = graph.get_tensor_by_name('image_tensor:0')
bbox = graph.get_tensor_by_name('detection_boxes:0')
classes = graph.get_tensor_by_name('detection_classes:0')

(outBoxes, classes) = sess.run([bbox, classes], feed_dict={imageTensor: np.expand_dims(image, axis=0)})

cnt = 0
imageHeight, imageWidth = image.shape[:2]
boxes = np.squeeze(outBoxes)
classes = np.squeeze(classes)
boxes = np.stack((boxes[:, 1] * imageWidth, boxes[:, 0] * imageHeight,
                  boxes[:, 3] * imageWidth, boxes[:, 2] * imageHeight),
                 axis=1).astype(np.int)

for i, bb, in enumerate(boxes):
    cv2.rectangle(image, (bb[0], bb[1]), (bb[2], bb[3]), (255, 255, 0), thickness=1)

# plt.figure(figsize=(10, 10))
# plt.imshow(image)
# plt.show()
cv2.imshow("Detected stuff", image)
cv2.waitKey(0)






