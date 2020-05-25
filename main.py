import cv2
import numpy as np
import math
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", default=0,
	help="path to optional input video file")
args = vars(ap.parse_args())

confThreshold = 0.7 
nmsThreshold = 0.4 
modelConfiguration = "assets/yolov3.cfg"
modelWeights = "assets/yolov3.weights"
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

# Human detection boxes
# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, detections):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIds = []
    confidences = []
    boxes = []
    resBoxes = []
    heights = []
    footmarks = []

    for out in detections:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold and classId == 0:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)

    for i in indices:
        i = i[0]
        box = boxes[i]
        left, top, width, height = box[0],  box[1], box[2], box[3]
        startX, startY, endX, endY = left, top, left+width, top+height
        footmarkX = left + width//2
        footmarkY = top + height//2
        heights.append(height)
        resBoxes.append([startX, startY, endX, endY])
        footmarks.append((footmarkX, footmarkY))

    if len(heights) == 0: heights = [0]

    return resBoxes, footmarks, int(np.average(heights))

# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def areClose(fm1, fm2, avg_height_px):
    # avg_height px => 1.7m
    # ? px => 2m
    safe_dist_m = 2
    avg_height_m = 2
    safe_dist_px = safe_dist_m * avg_height_px / avg_height_m

    return math.sqrt( (fm1[0]-fm2[0])**2 + (fm1[1]-fm2[1])**2 ) < safe_dist_px

vs = cv2.VideoCapture(args["input"])

while True:
    ret, frame = vs.read()
    if not ret: break

    frame = cv2.resize(frame, (920, 640))

    blob = cv2.dnn.blobFromImage(frame, 1/255, (448, 448), [0,0,0], 1, crop=False)
    net.setInput(blob)
    detections = net.forward(getOutputsNames(net))

    resBoxes, footmarks, avg_height = postprocess( frame, detections)
    numBoxes = len(resBoxes)
    isMarked = [False] * numBoxes


    for i in range(numBoxes):
        for j in range(i+1, numBoxes):
            if (not isMarked[i] or not isMarked[j]) and areClose(footmarks[i], footmarks[j], avg_height):
                isMarked[i] = True
                isMarked[j] = True

    for i, box in enumerate(resBoxes):
        (startX, startY, endX, endY) = box
        color = (0, 0, 255) if isMarked[i] else (0, 255, 0)
        frame = cv2.rectangle(frame, (startX, startY), (endX, endY),color, 2)
    

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(2)
    if key == ord('q'):
        break
