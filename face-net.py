# Emotion recognition with neural networks

import cv2
import numpy as np
from conx import *
import sys
import os
import time
import cProfile

INPUT_SIZE = 30
HIDDEN_LAYER_NODES = 25
OUTPUT_LAYER_NODES = 2
TRAINING_PERCENTAGE = 80
INPUTS_FILE = 'text_data/inputs.txt'
TARGETS_FILE = 'text_data/targets.txt'
WEIGHTS_FILE = 'text_data/weights.txt'
FACE_CLASSIFIER_FILE = 'text_data/haarcascade_frontalface.xml'
EYE_CLASSIFIER_FILE = 'text_data/haarcascade_eye.xml'
INPUT_FILENAMES_FILE = 'text_data/emotion_files.txt'
INPUT_IMAGE_DIR = "image_data/"

# main class for emotion recoginition
class EmotionRecognizer(Network):
    def __init__(self):
        Network.__init__(self)
        self.testInputs = []
        self.testTargets = []
        self.epsilon = 0.3
        self.momentum = 0.9
        self.reportRate = 1

    # establishes the correspondence between the targets and the labels
    def classify(self, output):
        assert len(output) == 2, 'invalid output pattern'
        if output[0] < self.tolerance:
            if output[1] < self.tolerance:
                return 'sad'
            else:
                return 'neutral'
        elif output[0] > (1 - self.tolerance):
            if output[1] < self.tolerance:
                return 'angry'
            else:
                return 'happy'
        else:
            return '???'

    # evaluate the current weights of the neural network
    def evaluate(self):
        if len(self.inputs) == 0:
            print 'no patterns to evaluate'
            return
        correct = 0
        wrong = 0
        for i in range(len(self.inputs)):
            pattern = self.inputs[i]
            target = self.targets[i]
            output = self.propagate(input=pattern)
            networkAnswer = self.classify(output)
            correctAnswer = self.classify(target)
            if networkAnswer == correctAnswer:
                correct = correct + 1
            else:
                wrong = wrong + 1
                print 'network classified image #%d (%s) as %s' % \
                      (i, correctAnswer, networkAnswer)
        total = len(self.inputs)
        correctPercent = float(correct) / total * 100
        wrongPercent = float(wrong) / total * 100
        print '%d patterns: %d correct (%.1f%%), %d wrong (%.1f%%)' % \
              (total, correct, correctPercent, wrong, wrongPercent)

    # splits the data in half for training and evaluation
    def splitData(self, trainingPortion=None):
        if type(trainingPortion) not in [int, float] or not 0 <= trainingPortion <= 100:
            print 'percentage of dataset to train on is required (0-100)'
            return
        patterns = zip(self.inputs + self.testInputs, self.targets + self.testTargets)
        assert len(patterns) > 0, "no dataset"
        print "Randomly shuffling data patterns..."
        random.shuffle(patterns)
        numTraining = int(math.ceil(trainingPortion / 100.0 * len(patterns)))
        self.inputs = [i for (i, t) in patterns[:numTraining]]
        self.targets = [t for (i, t) in patterns[:numTraining]]
        self.testInputs = [i for (i, t) in patterns[numTraining:]]
        self.testTargets = [t for (i, t) in patterns[numTraining:]]
        print "%d training patterns, %d test patterns" % (len(self.inputs), len(self.testInputs))

    # swap training and testing datasets to evaluate
    def swapData(self):
        print "Swapping training and testing sets..."
        self.inputs, self.testInputs = self.testInputs, self.inputs
        self.targets, self.testTargets = self.testTargets, self.targets
        self.showData()

    # print out the number of training images and test images
    def showData(self):
        print "%d training patterns, %d test patterns" % (len(self.inputs), len(self.testInputs))

# converts image files to floating point input for the neural net
def createInputData(directory, imagesFilenames, outputFile):
    out = open(outputFile, "w")
    names = open(imagesFilenames, "r")
    while 1:
        name = names.readline().strip()
        print directory+name
        if len(name) == 0: break
        image = cv2.imread(directory+name, 0)
        if image is None: continue
        w, h = image.shape
        image = np.reshape(image, (w*h))
        image = image/float(np.max(image))
        for value in image:
            out.write("%.4f " % value)
        out.write("\n")
    names.close()
    out.close()

# creates targets with the correct classification for each image
# assumes that filenames have the associated emotion in the filename
def createTargetData(imagesFilenames, outputFile):
    out = open(outputFile, "w")
    names = open(imagesFilenames, "r")
    while 1:
        name = names.readline().strip()
        if len(name) == 0: break
        if name.find("happy") != -1:
            out.write("1 1\n")
        elif name.find("sad") != -1:
            out.write("0 0\n")
        elif name.find("angry") != -1:
            out.write("1 0\n")
        else:
            out.write("0 1\n")
    names.close()
    out.close()

# creates both the target data and input data for the neural net
def generateData():
    createInputData(INPUT_IMAGE_DIR, INPUT_FILENAMES_FILE, INPUTS_FILE)
    createTargetData(INPUT_FILENAMES_FILE, TARGETS_FILE)        

# detects faces in images, crops them, and downsamples to INPUT_SIZE
# also saves the non-downsampled cropped version
def prepareImages(rootDir):
    face_cascade = cv2.CascadeClassifier(FACE_CLASSIFIER_FILE)
    eye_cascade = cv2.CascadeClassifier(EYE_CLASSIFIER_FILE)

    for root, dirs, files in os.walk(rootDir):
        for name in files:
            filename = os.path.join(root, name)
            if filename.find('crop') != -1: continue
            if filename.find('small') != -1: continue
            gray = cv2.imread(filename, 0)
            if gray is None: continue
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            if faces == (): continue
            (x,y,w,h) = faces[0]
            roi_gray = gray[y+(h*.2):y+h, x+(w*.1):x+w-(w*.1)]

            output = cv2.resize(roi_gray, (INPUT_SIZE, INPUT_SIZE))

            name = filename.split('.')[0]
            cv2.imwrite(name + '_crop.jpg', roi_gray)
            cv2.imwrite(name + '_small.jpg', output)

n = EmotionRecognizer()
n.addLayers(INPUT_SIZE**2, HIDDEN_LAYER_NODES, OUTPUT_LAYER_NODES)
n.loadInputsFromFile(INPUTS_FILE)
n.loadTargetsFromFile(TARGETS_FILE)
n.loadWeightsFromFile(WEIGHTS_FILE)
n.splitData(TRAINING_PERCENTAGE)

print "Emotion recognition network is set up"

# How to evaluate an image against the network
# Read in as grayscale
img = cv2.imread('happy-test.jpg', 0)

# find the face and crop it out
face_cascade = cv2.CascadeClassifier(FACE_CLASSIFIER_FILE)
faces = face_cascade.detectMultiScale(img, 1.3, 5)
(x,y,w,h) = faces[0]
roi = img[y+(h*.2):y+h, x+(w*.1):x+w-(w*.1)]

# resize and create input array of floats
img = cv2.resize(roi, (INPUT_SIZE, INPUT_SIZE))
img = np.reshape(img, (img.shape[0]*img.shape[1]))
img = img/float(np.max(img))

# Run through network and classify output
output = n.propagate(input=list(img))
print n.classify(output)






