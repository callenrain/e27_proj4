Repository Contents

face-net.py - main file for emotion recognition
conx.py - library file for the artificial neural network
image_data/ - contains all the image files used for training
text_data/ - contains text data for weights, targets, filenames
    emotion_files.txt - filenames of all the inputs to the neural net
    inputs.txt - images converted to floating point and concatenated
    targets.txt - targets for each of the images
    weights.txt - weights from the neural net after training
    haarcascade_frontalface.xml - HAAR weights for the face cascade classifier
    haarcascade_eye.xml - HAAR weights for the eye cascade classifier