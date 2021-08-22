import cv2
import numpy as np
import argparse

class yolo_fast_v2():
    def __init__(self, objThreshold=0.3, confThreshold=0.3, nmsThreshold=0.4):
        with open('coco.names', 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')   ###这个是在coco数据集上训练的模型做opencv部署的，如果你在自己的数据集上训练出的模型做opencv部署，那么需要修改self.classes
        self.stride = [16, 32]
        self.anchor_num = 3
        self.anchors = np.array([12.64, 19.39, 37.88, 51.48, 55.71, 138.31, 126.91, 78.23, 131.57, 214.55, 279.92, 258.87],
                           dtype=np.float32).reshape(len(self.stride), self.anchor_num, 2)
        self.inpWidth = 352
        self.inpHeight = 352
        self.net = cv2.dnn.readNet('model.onnx')
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.objThreshold = objThreshold
    def _make_grid(self, nx=20, ny=20):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)

    def postprocess(self, frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        ratioh, ratiow = frameHeight / self.inpHeight, frameWidth / self.inpWidth
        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        classIds = []
        confidences = []
        boxes = []
        for detection in outs:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > self.confThreshold and detection[4] > self.objThreshold:
                center_x = int(detection[0] * ratiow)
                center_y = int(detection[1] * ratioh)
                width = int(detection[2] * ratiow)
                height = int(detection[3] * ratioh)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                # confidences.append(float(confidence))
                confidences.append(float(confidence*detection[4]))
                boxes.append([left, top, width, height])

        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            frame = self.drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)
        return frame

    def drawPred(self, frame, classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=2)

        label = '%.2f' % conf
        label = '%s:%s' % (self.classes[classId], label)

        # Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        # cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine), (255,255,255), cv.FILLED)
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=1)
        return frame
    def detect(self, srcimg):
        blob = cv2.dnn.blobFromImage(srcimg, 1 / 255.0, (self.inpWidth, self.inpHeight))
        self.net.setInput(blob)
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())[0]

        outputs = np.zeros((outs.shape[0]*self.anchor_num, 5+len(self.classes)))
        row_ind = 0
        for i in range(len(self.stride)):
            h, w = int(self.inpHeight / self.stride[i]), int(self.inpWidth / self.stride[i])
            length = int(h * w)
            grid = self._make_grid(w, h)
            for j in range(self.anchor_num):
                top = row_ind+j*length
                left = 4*j
                outputs[top:top + length, 0:2] = (outs[row_ind:row_ind + length, left:left+2] * 2. - 0.5 + grid) * int(self.stride[i])
                outputs[top:top + length, 2:4] = (outs[row_ind:row_ind + length, left+2:left+4] * 2) ** 2 * np.repeat(self.anchors[i, j, :].reshape(1,-1), h * w, axis=0)
                outputs[top:top + length, 4] = outs[row_ind:row_ind + length, 4*self.anchor_num+j]
                outputs[top:top + length, 5:] = outs[row_ind:row_ind + length, 5*self.anchor_num:]
            row_ind += length
        return outputs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgpath', type=str, default='img/000139.jpg', help="image path")
    parser.add_argument('--objThreshold', default=0.3, type=float, help='object confidence')
    parser.add_argument('--confThreshold', default=0.3, type=float, help='class confidence')
    parser.add_argument('--nmsThreshold', default=0.4, type=float, help='nms iou thresh')
    args = parser.parse_args()

    srcimg = cv2.imread(args.imgpath)
    model = yolo_fast_v2(objThreshold=args.objThreshold, confThreshold=args.confThreshold, nmsThreshold=args.nmsThreshold)
    outputs = model.detect(srcimg)
    srcimg = model.postprocess(srcimg, outputs)

    winName = 'Deep learning object detection in OpenCV'
    cv2.namedWindow(winName, 0)
    cv2.imshow(winName, srcimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()