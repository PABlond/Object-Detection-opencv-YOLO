
import numpy as np
import cv2
import argparse
from random import randint
from darkflow.net.build import TFNet
import os
import wget


class Detect:
    def __init__(self, input_path):
        self.cfg_file = "./cfg/yolo.cfg"
        self.weights_file = "./cfg/yolo.weights"
        self.init()
        self.input = input_path
        self.dectected_objs = {}
        self.boxes = []
        self.crop = []
        self.colors = {}
        self.frame = None
        self.out = None
        self.ctrler = None

    def init(self):
        # Download model data
        if not os.path.exists(self.weights_file):
            print('Downloading {}'.format(self.weights_file))
            wget.download(
                'https://s3.amazonaws.com/evopter/yolo.weights', self.weights_file)
        # Import the model
        options = {"model": self.cfg_file,
                   "load": self.weights_file, "threshold": 0.1}
        self.tfnet = TFNet(options)

    def check_label_color(self, label):
        if not label in self.colors:
            self.colors[label] = [
                randint(0, 255), randint(0, 255), randint(0, 255)]

    def draw_obj_info(self):
        for box in self.boxes:
            if box['confidence'] > 0.5:
                x, y, x1, y1, label, confidence = box['topleft']['x'], box['topleft']['y'], box[
                    'bottomright']['x'], box['bottomright']['y'], box['label'], box['confidence']
                w, h = x1 - x, y1 - y

                cv2.rectangle(self.frame, (x1, y), (x1+35*len(label),
                                                    y+35), (255, 255, 255), cv2.FILLED)
                cv2.putText(self.frame, label, (x1, y+30),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, self.colors[label])

                cv2.rectangle(self.frame, (x1, y+45), (x1+35*len(str(int(confidence*100)
                                                                     ) + '%'), y+90), (255, 255, 255), cv2.FILLED)
                cv2.putText(self.frame, str(int(confidence*100)) + '%',
                            (x1, y+90), cv2.FONT_HERSHEY_SIMPLEX, 2, self.colors[label])

    def draw_ctrler_boxes(self):
        x_offset = 0
        y_offset = 150
        i = 0
        for crop_img in self.crop:
            if i < 6:
                s_img, l_img = crop_img, self.ctrler
                l_img[y_offset:y_offset+s_img.shape[0],
                      int(x_offset): int(x_offset+s_img.shape[1])] = s_img
                self.ctrler = l_img
                x_offset += self.frame_width / 6
                i += 1

    def draw_boxes(self):
        crop_imgs = []
        for box in self.boxes:
            if box['confidence'] > 0.5:
                x, y, x1, y1, label = box['topleft']['x'], box['topleft']['y'], box[
                    'bottomright']['x'], box['bottomright']['y'], box['label']
                w, h = x1 - x, y1 - y
                # If needed, create a new color for this label
                self.check_label_color(label)
                # Draw rect around the object
                cv2.rectangle(self.frame, (x, y), (x1, y1),
                              self.colors[label], 1)
                # Get the object into a new frame
                crop_img = self.frame[y+1:y+h-1, x+1:x+w-1]
                # If needed, resize the object to match the ctrler frame
                if crop_img.shape[1] > self.frame_width / 6 or crop_img.shape[1] < 100:
                    crop_img = cv2.resize(crop_img, (int(self.frame_width / 6),
                                                     int(crop_img.shape[0] * self.frame_width / 6 / crop_img.shape[1])))
                if self.frame_height < crop_img.shape[0] + 150:
                    crop_img = cv2.resize(
                        crop_img, (int((self.frame_height - (self.frame_height - 150)) * crop_img.shape[1] / self.frame_width / 6), int(self.frame_height - (self.frame_height - 150))))

                crop_imgs.append(crop_img)

        self.draw_obj_info()
        self.draw_ctrler_boxes()
        self.crop = crop_imgs

    def build(self, output_path):
        cap = cv2.VideoCapture(self.input)
        # Get stream dimensions
        frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
        self.frame_width, self.frame_height = frame_width, frame_height
        frame_fps = cap.get(cv2.CAP_PROP_FPS)

        self.out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(
            'M', 'J', 'P', 'G'), frame_fps, (frame_width*2, frame_height*2))

        if (cap.isOpened() == False):
            print("Error opening video stream or file")
        # Read until video is completed
        while(cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()

            self.frame = frame
            original_frame = frame.copy()
            gray_frame = cv2.Laplacian(original_frame.copy(), cv2.CV_8U)

            # Get the frame index as timestamp
            timestamp = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            print(timestamp)

            if ret == True:
                # Init the controler frame
                self.ctrler = np.zeros(
                    (frame_height, frame_width, 3), np.uint8)
                # Make a prediction
                self.boxes, self.dectected_objs = self.tfnet.return_predict(self.frame), {
                }
                self.draw_boxes()
                # Organize UI
                left_frame = np.vstack((original_frame, gray_frame))
                right_frame = np.vstack((self.frame, self.ctrler))
                main_frame = np.hstack((left_frame, right_frame))
                # Append the final frame to the output video
                self.out.write(main_frame)
                # Check the final frame
                cv2.imshow('Output', main_frame)
                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            # Exit if the stream is over
            else:
                break
        # Finally
        cap.release()
        self.out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Get input and output stream
    ap = argparse.ArgumentParser(
        description='Object detection with Opencv & the yolo model')
    ap.add_argument("-i", "--inp", type=str, default='data/input/sample.mp4',
                    help="input file to detect from")
    ap.add_argument("-o", "--out", type=str, default='data/output/sample.avi',
                    help="output file to create")
    args = ap.parse_args()
    # Make a prediction and generate an output file
    Detect(args.inp).build(args.out)
