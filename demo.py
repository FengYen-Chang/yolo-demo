import sys
import time

sys.path.append('./')

from yolo.net.yolo_tiny_net import YoloTinyNet 
import tensorflow as tf 
import cv2
import numpy as np

classes_name =  ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]

# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
  # if there are no boxes, return an empty list
  if len(boxes) == 0:
    return []
 
  # if the bounding boxes integers, convert them to floats --
  # this is important since we'll be doing a bunch of divisions
  if boxes.dtype.kind == "i":
    boxes = boxes.astype("float")
 
  # initialize the list of picked indexes 
  pick = []
 
  # grab the coordinates of the bounding boxes
  x1 = boxes[:,0]
  y1 = boxes[:,1]
  x2 = boxes[:,2]
  y2 = boxes[:,3]
 
  # compute the area of the bounding boxes and sort the bounding
  # boxes by the bottom-right y-coordinate of the bounding box
  area = (x2 - x1 + 1) * (y2 - y1 + 1)
  idxs = np.argsort(y2)
 
  # keep looping while some indexes still remain in the indexes
  # list
  while len(idxs) > 0:
    # grab the last index in the indexes list and add the
    # index value to the list of picked indexes
    last = len(idxs) - 1
    i = idxs[last]
    pick.append(i)
 
    # find the largest (x, y) coordinates for the start of
    # the bounding box and the smallest (x, y) coordinates
    # for the end of the bounding box
    xx1 = np.maximum(x1[i], x1[idxs[:last]])
    yy1 = np.maximum(y1[i], y1[idxs[:last]])
    xx2 = np.minimum(x2[i], x2[idxs[:last]])
    yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)
 
    # compute the ratio of overlap
    overlap = (w * h) / area[idxs[:last]]
 
    # delete all indexes from the index list that have
    idxs = np.delete(idxs, np.concatenate(([last],
      np.where(overlap > overlapThresh)[0])))
 
  # return only the bounding boxes that were picked using the
  # integer data type
  return boxes[pick].astype("int")

def process_predicts(predicts):
  p_classes = predicts[0, :, :, 0:20]
  #print predicts.shape
  #print p_classes.shape
  C = predicts[0, :, :, 20:22]
  #print C.shape
  coordinate = predicts[0, :, :, 22:]

  p_classes = np.reshape(p_classes, (7, 7, 1, 20))
  C = np.reshape(C, (7, 7, 2, 1))

  P = C * p_classes

  #print P[5,1, 0, :]
  #print np.argmax(p_classes), p_classes.shape

  #print P
  
  index = np.argmax(P)
  #print np.argmax(P), P.shape 
  index = np.unravel_index(index, P.shape)
  #print index, P[index]
  
  a, b, c, num_cls = np.where(P > 0.05)

  print a, b, c, num_cls

  #print P[:, :, :, index[3]].argmax()
  #print P.argmax()

  class_num = index[3]

  coordinate = np.reshape(coordinate, (7, 7, 2, 4))

  max_coordinate = coordinate[index[0], index[1], index[2], :]
  print max_coordinate

  xcenter = max_coordinate[0]
  ycenter = max_coordinate[1]
  w = max_coordinate[2]
  h = max_coordinate[3]

  xcenter = (index[1] + xcenter) * (448/7.0)
  ycenter = (index[0] + ycenter) * (448/7.0)

  w = w * 448
  h = h * 448

  xmin = xcenter - w/2.0
  ymin = ycenter - h/2.0

  xmax = xmin + w
  ymax = ymin + h

  scale = 448 / 7.0
  """
  sel_coordinate = coordinate[a, b, c, :]
  print sel_coordinate
  _xc, _yc, _w, _h = sel_coordinate
  """
  _xc, _yc, _w, _h = coordinate[a, b, c, 0], coordinate[a, b, c, 1], coordinate[a, b, c, 2], coordinate[a, b, c, 3]

  _xc[:] = (b[:] + _xc[:]) * scale
  _yc[:] = (a[:] + _yc[:]) * scale
  _w[:] = _w[:] * 448
  _h[:] = _h[:] * 448

  x_min = _xc[:] - _w[:] / 2.0
  y_min = _yc[:] - _h[:] / 2.0

  x_max = x_min + _w
  y_max = y_min + _h

  #print len(_xc)
  rect = np.zeros((len(_xc), 4))
  rect[:, 0], rect[:, 1], rect[:, 2], rect[:, 3] = x_min.astype(int), y_min.astype(int), x_max.astype(int), y_max.astype(int)

  con_rect = np.ascontiguousarray(rect).view(np.dtype((np.void, rect.dtype.itemsize * rect.shape[1])))
  _, idx = np.unique(con_rect, return_index=True)
  
  u_rect = rect[idx]
  u_class = idx
  print u_class

  for i in range(len(u_rect)):
    position = np.where(rect[:, 1] == u_rect[i, 1])
    print position, position[0].shape
    if position[0].shape[0] > 1 : 
      temp = np.zeros(position[0].shape[0])
      for j in range(position[0].shape[0]) : 
        temp[j] = P[a[position[0][j]], b[position[0][j]], c[position[0][j]], num_cls[position[0][j]]]
      print temp
      index_temp = np.argmax(temp)
      index_temp = np.unravel_index(index_temp, temp.shape)
      print index_temp
      u_class[i] = position[0][index_temp]

  print u_class

  print u_rect, u_rect.shape

  boundingBox = non_max_suppression_fast(u_rect, 0.5)
  print boundingBox, boundingBox.shape

  size = len(boundingBox)

  cls_set = np.zeros(size)
  for i in range(size) :
    cls_set[i] = num_cls[u_class[np.where(u_rect[:, 1] == boundingBox[i, 1])[0]]].astype(int)
    
  print cls_set

  return xmin, ymin, xmax, ymax, class_num, boundingBox, rect, cls_set

common_params = {'image_size': 448, 'num_classes': 20, 
                'batch_size':1}
net_params = {'cell_size': 7, 'boxes_per_cell':2, 'weight_decay': 0.0005}

net = YoloTinyNet(common_params, net_params, test=True)

image = tf.placeholder(tf.float32, (1, 448, 448, 3))
predicts = net.inference(image)

sess = tf.Session()

np_img = cv2.imread('bus_human.jpg')
resized_img = cv2.resize(np_img, (448, 448))
np_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)


np_img = np_img.astype(np.float32)

np_img = np_img / 255.0 * 2 - 1
np_img = np.reshape(np_img, (1, 448, 448, 3))

saver = tf.train.Saver(net.trainable_collection)

saver.restore(sess, 'models/pretrain/yolo_tiny.ckpt')

np_predict = sess.run(predicts, feed_dict={image: np_img})

start = time.time()
xmin, ymin, xmax, ymax, class_num, boundingBox, rect, cls_set = process_predicts(np_predict)
end = time.time()
print 'time = ', end - start

np_img_1 = cv2.imread('bus_human.jpg')
temp_resize_img = cv2.resize(np_img_1, (448, 448))
print xmin, ymin, xmax, ymax

np_img_2 = cv2.imread('bus_human.jpg')
temp_resize_img_2 = cv2.resize(np_img_2, (448, 448))

class_name = classes_name[class_num]
cv2.rectangle(resized_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255))
cv2.putText(resized_img, class_name, (int(xmin), int(ymin)), 2, 1.5, (0, 0, 255))
cv2.imwrite('cat_out.jpg', resized_img)

#print len(boundingBox)
for i in range(len(boundingBox)) :
  cv2.rectangle(temp_resize_img, (boundingBox[i, 0], boundingBox[i, 1]), (boundingBox[i, 2], boundingBox[i, 3]), (0, 0, 255))
  cv2.putText(temp_resize_img, classes_name[int(cls_set[i])], (boundingBox[i, 0], boundingBox[i, 1]), 2, 1.5, (0, 0, 255))
cv2.imwrite('cat_out1.jpg', temp_resize_img)

for i in range(len(rect)) :
  cv2.rectangle(temp_resize_img_2, (int(rect[i, 0]), int(rect[i, 1])), (int(rect[i, 2]), int(rect[i, 3])), (0, 0, 255))
  #cv2.putText(temp_resize_img, class_name, (boundingBox[i, 0], boundingBox[i, 1]), 2, 1.5, (0, 0, 255))
cv2.imwrite('cat_out2.jpg', temp_resize_img_2)


sess.close()
