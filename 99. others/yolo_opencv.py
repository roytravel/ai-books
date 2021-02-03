# -*- coding:utf-8 -*-
import cv2
import numpy as np

def get_shape():
    ## Read the Web Camera
    # ret, frame = VideoSignal.read()
    # h, w, c = frame.shape

    # Read the figure
    image = cv2.imread('cat.jpg')
    height, width, channel = image.shape

    return image, height, width, channel


def create_network():
    
    # load YOLO weight and cfg file
    YOLO_net = cv2.dnn.readNet("yolov3-spp.weights","yolov3-spp.cfg")

    # reconstruct the YOOLO network
    classes = []

    # read the class list of object
    with open("yolo.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
        layer_names = YOLO_net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in YOLO_net.getUnconnectedOutLayers()]

    return YOLO_net, classes, output_layers


def network_io(image):
    # Input image into YOLO
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    # blob = cv2.dnn.blobFromImage(image, 0.00392, (320, 320), (0, 0, 0), True, crop=False)

    # Set the new input value for the network
    YOLO_net.setInput(blob)

    # Output from YOLO network
    outputs = YOLO_net.forward(output_layers)

    return outputs


if __name__ == '__main__':
    
    # get signal from web camera
    VideoSignal = cv2.VideoCapture(0)    
    
    # create network
    YOLO_net, classes, output_layers = create_network()

    while True:
        
        # get the image, height, width and channel
        image, height, width, channel = get_shape()

        # variables for inserting detected object data
        class_ids = []
        confidences = []
        boxes = []

        # input the image into network and get output
        outputs = network_io(image)
        
        for output in outputs:

            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)

                    dw = int(detection[2] * width)
                    dh = int(detection[3] * height)

                    # Rectangle coordinate
                    x = int(center_x - dw / 2)
                    y = int(center_y - dh / 2)
                    boxes.append([x, y, dw, dh])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Non-Maximum Suppression for removing false positive and getting final prediction
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.45, 0.4)

        for idx in range(len(boxes)):
            if idx in indexes:
                x, y, w, h = boxes[idx]
                label = str(classes[class_ids[idx]])
                score = confidences[idx]

                # Input the bounding box and class information into image
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 5)
                cv2.putText(image, label, (x, y - 20), cv2.FONT_ITALIC, 0.5, (255, 255, 255), 1)

        cv2.imshow("YOLOv3", image)

        if cv2.waitKey(100) > 0:
            break
