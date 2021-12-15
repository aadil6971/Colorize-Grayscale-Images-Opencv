
# import the necessary packages
import numpy as np
import argparse
import cv2

prototxt="model/colorization_deploy_v2.prototxt"
caffe = "model/colorization_release_v2.caffemodel"
points = "model/pts_in_hull.npy"
image_path = "images/test10.jpg"

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt, caffe)
pts = np.load(points)


class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]


image = cv2.imread(image_path)
scaled = image.astype("float32") / 255.0
lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

resized = cv2.resize(lab, (224, 224))
L = cv2.split(resized)[0]
L -= 50


'print("[INFO] colorizing image...")'
net.setInput(cv2.dnn.blobFromImage(L))
ab = net.forward()[0, :, :, :].transpose((1, 2, 0))


ab = cv2.resize(ab, (image.shape[1], image.shape[0]))


L = cv2.split(lab)[0]
colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)


colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
colorized = np.clip(colorized, 0, 1)


colorized = (255 * colorized).astype("uint8")

# show the original and output colorized images
cv2.imshow("Original", image)
cv2.imshow("Colorized", colorized)
cv2.waitKey(0)
cv2.imwrite(image_path+"edited.jpg",colorized)