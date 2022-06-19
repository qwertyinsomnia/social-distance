import cv2
import onnxruntime as ort
import numpy as np
import argparse

class PoseDet():
    def __init__(self):
        self.inpWidth = 256
        self.inpHeight = 256
        self.line_width = 2
        self.keep_ratio = True
        self.swaprgb = False

        self.COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")

        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.model = 'pose_landmark_heavy.onnx'
        self.net = ort.InferenceSession(self.model, so, providers=['CPUExecutionProvider']) # CUDAExecutionProvider


    def resize_image(self, srcimg):
        top, left, newh, neww = 0, 0, self.inpHeight, self.inpWidth
        if self.keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = self.inpHeight, int(self.inpWidth / hw_scale)
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                left = int((self.inpWidth - neww) * 0.5)
                img = cv2.copyMakeBorder(img, 0, 0, left, self.inpWidth - neww - left, cv2.BORDER_CONSTANT,
                                         value=0)  # add border
            else:
                newh, neww = int(self.inpHeight * hw_scale), self.inpWidth
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                top = int((self.inpHeight - newh) * 0.5)
                img = cv2.copyMakeBorder(img, top, self.inpHeight - newh - top, 0, 0, cv2.BORDER_CONSTANT, value=0)
        else:
            img = cv2.resize(srcimg, (self.inpWidth, self.inpHeight), interpolation=cv2.INTER_AREA)
        return img, newh, neww, top, left


    def detect(self, frame):
        srcimg = frame.copy() # copy so origin image will not resized
        if self.swaprgb:
            srcimg = cv2.cvtColor(srcimg, cv2.COLOR_BGR2RGB)
        img, newh, neww, top, left = self.resize_image(srcimg) # resize into 256x256
        # print(img.shape)
        blob = np.expand_dims(np.transpose(img, (0, 1, 2)), axis=0).astype(np.float32) / 255.0
        outs = self.net.run(None, {self.net.get_inputs()[0].name: blob})[0].squeeze(axis=0)

        srcHeight, srcWidth, _ = srcimg.shape # 256, 256, _
        # srcHeight, srcWidth, _ = img.shape

        if self.keep_ratio:
            img, landmarks = self.draw_landmarks_with_border(frame, outs, srcHeight, srcWidth, neww, left)
        else:
            img = self.draw_landmarks_without_border(frame, outs, srcHeight, srcWidth)
        

        # cv2.imshow("winName", img)
        # cv2.waitKey(0)
        return landmarks


    def draw_landmarks_without_border(self, image, outs, srcHeight, srcWidth):
        for j in (0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28): # 33
            point = (int(outs[5*j] / self.inpWidth * srcWidth), int(outs[5*j + 1] / self.inpHeight * srcHeight))
            # print(point)
            cv2.circle(image, point, radius=0, color=(0, 0, 255), thickness=5)
        for j in ((11, 13), (13, 15), (12, 14), (14, 16), (11, 12), (23, 25), (25, 27), (24, 26), (26, 28), (11, 23), (23, 24), (12, 24)): # 11 17
            point1 = (int(outs[5*j[0]] / self.inpWidth * srcWidth), int(outs[5*j[0] + 1] / self.inpHeight * srcHeight))
            point2 = (int(outs[5*j[1]] / self.inpWidth * srcWidth), int(outs[5*j[1] + 1] / self.inpHeight * srcHeight))
            color = [int(c) for c in self.COLORS[j[1] % len(self.COLORS)]]
            cv2.line(image, point1, point2, color=(0, 0, 255), thickness=self.line_width)
        neckpoint = (int((outs[5*11] + outs[5*12]) / 2 / self.inpWidth * srcWidth), int((outs[5*11+1] + outs[5*12+1]) / 2 / self.inpHeight * srcHeight))
        nosepoint = (int(outs[5*0] / self.inpWidth * srcWidth), int(outs[5*0 + 1] / self.inpHeight * srcHeight))
        cv2.circle(image, neckpoint, radius=0, color=(0, 0, 255), thickness=5)
        cv2.line(image, neckpoint, nosepoint, color=(0, 0, 255), thickness=self.line_width)

        return image


    def draw_landmarks_with_border(self, image, outs, srcHeight, srcWidth, neww, left):
        landmarks = []
        for j in (0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28): # 33
            point = (int((outs[5*j] - left) / neww * srcWidth), int(outs[5*j + 1] / self.inpHeight * srcHeight))
            # print(point)
            landmarks.append(point)
            cv2.circle(image, point, radius=0, color=(0, 0, 255), thickness=5)
        for j in ((11, 13), (13, 15), (12, 14), (14, 16), (11, 12), (23, 25), (25, 27), (24, 26), (26, 28), (11, 23), (23, 24), (12, 24)): # 11 17
            point1 = (int((outs[5*j[0]] - left) / neww * srcWidth), int(outs[5*j[0] + 1] / self.inpHeight * srcHeight))
            point2 = (int((outs[5*j[1]] - left) / neww * srcWidth), int(outs[5*j[1] + 1] / self.inpHeight * srcHeight))
            color = [int(c) for c in self.COLORS[j[1] % len(self.COLORS)]]
            cv2.line(image, point1, point2, color=(0, 0, 255), thickness=self.line_width)
        neckpoint = (int(((outs[5*11] + outs[5*12]) / 2 - left) / neww * srcWidth), int((outs[5*11+1] + outs[5*12+1]) / 2 / self.inpHeight * srcHeight))
        nosepoint = (int((outs[5*0] - left) / neww * srcWidth), int(outs[5*0 + 1] / self.inpHeight * srcHeight))
        cv2.circle(image, neckpoint, radius=0, color=(0, 0, 255), thickness=5)
        cv2.line(image, neckpoint, nosepoint, color=(0, 0, 255), thickness=self.line_width)

        return image, landmarks