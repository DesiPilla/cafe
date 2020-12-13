import os
import re
import cv2
import glob
import json
import random
import shutil
import operator
import requests

import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from imutils import paths
from shutil import copyfile

# Import matplotlib modules
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Import sklearn modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, auc, roc_curve

# Import tensorflow and keras modules
import tensorflow as tf

import keras
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Sequential
from keras.layers import Activation,Conv2D, Dense, Dropout, Flatten
from keras.layers import GlobalAveragePooling2D, Input, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping

import keras_vggface
from mtcnn import MTCNN

from matplotlib.cm import get_cmap
from keras import backend as K
from tensorflow.python.ops import gen_nn_ops

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')


# Define class label names
class_labels = {0: "sip!",
                1: "skip :("}
                           
# Define all model names
model_names = ["google1", "google2", "google3", "google4", "google5"]

# Define the optimal threshold for each model
model_threshes = {
    "google1": 0.28,
    "google2": 0.44,
    "google3": 0.46,
    "google4": 0.29,
    "google5": 0.37
}



def load_image(path):
    '''
    This function loads an image into a CV2 image from a specified path.
    The image is also converted from BGR to RGB.

    Inputs:
    -------------------------
    path: str, the local filepath of the the image


    Returns:
    -------------------------
    img: cv2 image / 3D array

    '''
    img = cv2.imread(path)                      # Read in image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    return img


def display_many(paths, num_to_plot=25, results=[None, None], randomize=False):
    '''
    This function displays many images in a grid-like pattern on a single figure.

    Inputs:
    -------------------------
    paths: [str, str, ...], list of filepaths containing images to display
    num_to_plot: int (default = 25)
        * the number of images to display in the figure
        * multiples of 5 work best, as the figure will always have 5 columns
    results: [scores, classes]
        * scores: 
            ** list of sip scores corresponding to the images located in the
               `paths` parameter.
            ** scores are the first element of the output of the keras models
            ** ex. [0.56, 0.40, 0.97, 0.22, ...]
        * classes:
            ** list of the true labels corresponding to the images located in the
               `paths` parameter/
            ** ex. ['sip!', 'skip :(', 'sip !', 'skip :(', ...]
    randomize: boolean (default = False)
        * whether or not to randomize the files before taking a subset of size
          `num_to_plot` to plot


    Returns:
    -------------------------
    None (figure is displayed in-function using fig.show())
    
    '''
    # Assert number of images to plot is not too large
    if num_to_plot > len(paths):
        num_to_plot = len(paths)

    # Calculate the number of rows needed for the plot
    n_cols = 5
    n_rows = num_to_plot // n_cols
    row_height = 3
    if results[0] is not None:
        row_height = 4
    
    if randomize:
        # Randomly select num_to_plot image paths
        idx = np.arange(len(paths))
        idx = np.random.choice(idx, num_to_plot)
        paths = np.array(paths)[idx]

        # If results are provided, get the corresponding values
        if results[0] is not None:
            scores = results[0][idx]
            classes = results[1][idx]
    else:
        # Select the first num_to_plot image paths
        paths = paths[:num_to_plot]

        # If results are provided, get the corresponding values
        if results[0] is not None:
            scores = results[0][:num_to_plot]
            classes = results[1][:num_to_plot]

    # Load the images and resize to (256, 256) with padding
    imgs = [load_image(p) for p in paths]
    imgs = [resize_with_pad(img, 256) for img in imgs]

    # Plot the images
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(15, row_height*n_rows))
    for i, axes in enumerate(ax.flatten()):
        axes.imshow(imgs[i])
        axes.set_yticks([])     # Remove yticks
        axes.set_xticks([])     # Remove xticks

        # If results are provided, display them in the title
        if results[0] is not None:
            axes.set_title("Sip match: {:.2%}\nActual: {}".format(scores[i], class_labels[classes[i]]))
    
    # Format the whitespace between images
    if results[0] is not None: fig.subplots_adjust(wspace=0, hspace=0.3)
    else:          
        fig.tight_layout()
        fig.subplots_adjust(wspace=0, hspace=0)

    # Show the figure
    fig.show()
    return


def resize_with_pad(img, padded_size):
    '''
    This function takes an image and scales it to a square image with 
    height = width = padded_size. 
    
    The larger dimension of the original image is scaled to padded_size while
    maintaining the original aspect ratio. The scaled image is then padded such
    that the smaller dimension is now also `padded_size`. 
    
    The padded pixels are black (0, 0, 0).

    Inputs:
    -------------------------
    img: cv2 image / 2D array / 3D array
    padded_size: int
        * the number of pixels of the new height and width of the scaled image

    Returns:
    -------------------------
    new_img: cv2 image / 2D array / 3D array
    
    '''
    # Get height and width of the original image
    height, width = img.shape[:2]

    # Find the ratio between padded_size and the largest dimension
    ratio = padded_size / max(height, width)

    # Scale up the width and height while maintaining the original aspect ratio
    padded_width = int(width * ratio)
    padded_height = int(height * ratio)

    # Resize the image
    img = cv2.resize(img, (padded_width, padded_height))

    # Calculate the padding required to make the scaled image a square
    delta_w = padded_size - padded_width
    delta_h = padded_size - padded_height
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    # Pad the image
    color = [0, 0, 0]
    new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)
    return new_img


def get_face(img):
    '''
    This function takes an image and locates all faces present. 
    One face is selected, and the image is cropped to just include that face. 
    The cropped image is scaled to (256, 256) with padding.
    
    The cropped image and the "patch" of the face location in the original image
    is returned.

    If no faces were found, the original image and None are returned.

    This function is based off code from 
    https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/

    Inputs:
    -------------------------
    img: cv2 image / 2D array / 3D array

    Returns:
    -------------------------
    square_face: cv2 image / 2D array / 3D array
        * the cropped image containing the face (padded to (256, 256))
        * if no faces were found, the original image is returned

    face_patch: matplotlib.patches.Rectangle
        * the location of the returned face in the original image
        * if no faces were found, None is returned 
    
    '''
    # Create the detector, using default weights
    detector = MTCNN()

    # Detect faces in the image
    results = detector.detect_faces(img)

    # Extract the bounding box from the first face
    try:
        x1, y1, width, height = results[0]['box']
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2, y2 = x1 + width, y1 + height
    
    # If no faces are found, return the original image and None
    except: 
        return img, None

    # Extract the face
    face = img[y1:y2, x1:x2]
    face_patch = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r',facecolor='none')
    
    # Resize and pad the face to be a 256x256 square image
    square_face = resize_with_pad(face, 256)

    return square_face, face_patch


def plot_with_box(img, patch):
    '''
    This function plots an image and patch on the same figure.

    Inputs:
    -------------------------
    imgs: cv2 image / 2D array / 3D array
    patches: matplotlib.patches.Rectangle object

    Returns:
    -------------------------
    fig: matplotlib.pyplot.figure
    '''

    fig, ax = plt.subplots(1)   # Create figure
    ax.imshow(img)              # Display image
    ax.add_patch(patch)         # Add patch
    fig.show()                  # Show figure

    return fig

def plot_multiple_with_box(imgs, face_patches):
    '''
    This function displays all images and their corresponding patches in a 
    grid-like pattern on the same figure.

    Inputs:
    -------------------------
    imgs: list of cv2 images / 2D arrays / 3D arrays
    patches: list of matplotlib.patches.Rectangle objects

    Returns:
    -------------------------
    fig: matplotlib.pyplot.figure
    '''
    # Calculate the number of rows needed for the plot
    n_cols = min(5, len(imgs))
    n_rows = max(1, len(imgs) // n_cols)

    # Plot the images and face patches
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(25, 7*n_rows))
    for i, axes in enumerate(ax.flatten()):
        axes.imshow(imgs[i])            # Display image
        axes.add_patch(face_patches[i])      # Add face patch
        axes.set_yticks([])             # Remove yticks
        axes.set_xticks([])             # Remove xticks
    
    # Show fig
    fig.show()
    return fig


def get_urls(path):
    '''
    This function reads a .txt file containing URLs and returns them as a list.

    Inputs:
    -------------------------
    path: str, path to .txt file containing URLs

    Returns:
    -------------------------
    urls: list of URLs
    '''
    with open(path, 'rb') as f:
        urls = f.read().decode("utf-8").split('\n')
    return urls


def print_search_term_balance(file_names):
    '''
    This function prints out the balance of the search terms in a list of file 
    names. The file names are stripped down to just their search term (which 
    precedes the file_id) and the balance is then calculated.

    Inputs:
    -------------------------
    file_names: list of str
        * list of all file_names to be included in the balance calculations

    Returns:
    -------------------------
    None
    '''
    # Count the number of files belonging to each search term
    search_terms = np.array([f.split('00')[0].split('/')[-1][:-1] for f in file_names])
    terms, counts = np.unique(search_terms, return_counts=True)
    idx = np.argsort(-counts)
    for t, c in zip(terms[idx], counts[idx]):
        print('{:25.25s}: {:} photos ({:.2%})'.format(t, c, c / counts.sum()))
    print("-----------------------------------------------")
    print("Total: {} images".format(counts.sum()))


def print_sip_skip_balance(sip_files, skip_files, col='senti'):
    '''
    This function prints out the balance between two directories.

    Inputs:
    -------------------------
    sip_files: list of str
        * list of all files in the `sip` directory
    skip_files: list of str
        * list of all files in the `skip` directory

    Returns:
    -------------------------
    counts: list of int
        * list containing the number of sip files and the number of skip files
        * ex. [520, 667]
    perc: list of float
        * the `counts` list, but normalized by dividing the total number of files
          in both directories
        * ex. [0.43808, 0.56192]
    '''
    labels = ("sip", "skip")
    counts = np.array([len(sip_files), len(skip_files)])
    perc = dict(zip(labels, counts / counts.sum()))

    for k in labels:
        print('This dataset contains {0} {1} images ({2:.1%})'.format(int(perc[k]*counts.sum()), k, perc[k]))    
    return counts, perc


def download_images(urls, dir='.', label='', num=None):
    '''
    This function fetches images located at the provided URLs and saves 
    them locally.

    If the file cannot be loaded as a cv2 image, then there is something wrong 
    with the file and it is deleted from local memory.

    Inputs:
    -------------------------
    urls: list of str
        * list of URLs where images can be fetched
    dir: str, (default = '.')
        * path to directory where downloaded images should be saved
    label: str, (default = '')
        * label for all images in this set
        * the label will precede the file_id in the file name
        * ex. 'young_woman', 'young_woman_black', etc.
    num: int, (default = None)
        * number of urls to download from the provided list
        * if None, all urls will be downloaded
        * if a value, the first num urls will be downloaded. The list is not
          randomized prior to truncating.

    Returns:
    -------------------------
    None
    '''
    # Select number of urls to download
    if num is not None:
        urls = urls[:num]

    # Loop over all urls in the given list
    for i, url in enumerate(urls):
        try:
            # Download the image
            r = requests.get(url, timeout=60)
            
            # Save the image to the specified directory
            path = "{}/{}_{}.jpg".format(dir, label, str(i).zfill(6))
            f = open(path, "wb")
            f.write(r.content)
            f.close()
            
            # Try to load the image
            img = load_image(path)

            # Print updates
            if not (i+1) % 25: print("[INFO] downloaded: {}/{}\t{}".format(i+1, len(urls), path))

        # Handle any exceptions are thrown during the download process
        except:
            print("[INFO] error downloading {}/{}\t{}...skipping".format(i+1, len(urls), path))
            try:
               os.remove(urls[i])
            except:
                pass
    print("Done.")


def load_image_set(file_set, face_only=True, verbose=True):
    '''
    This function will load many images. The user can input the filepath to a
    directory containing many images, or a list of filepaths to load.

    The user can also specify whether the images are to be loaded as-is, or if
    they should be scanned for faces and cropped to one.


    Inputs:
    -------------------------
    file_set: directory containing images --OR-- list of image filepaths
        * if directory: str, path to a dircetory containing the images to load
        * if list: list of str, where each element is a filepath of an image to load
    
    face_only: boolean (default = True)
        * if True, each image will be parsed by get_face() after being loaded
          and will be replaced by the returned image.

    vebose: Boolean, (default = True)
        * whether or not to pring out progress of function

    Returns:
    -------------------------
    images: list of cv2 images / 2D arrays / 3D arrays
    '''
    # If True, file_set is the path to the directory containing the images
    if type(file_set) == str:
        # Get all filepaths within the directory
        files = glob.glob(file_set + '/*.jpg')

    # If False, file_set is a list of file_paths for the images
    else:
        files = file_set
    
    files.sort()

    if verbose: print("Loading {:,} images...".format(len(files)))
    images = []
    for i in range(len(files)):
        if verbose and not (i+1)%10: print("\tLoading image {}/{}".format(i+1, len(files)))
        img = load_image(files[i])          # Load image
        if not face_only:       
            images.append(img)              # Add original image (if face_only = False)
        else:
            images.append(get_face(img)[0]) # Add cropped image (if face_only = True)

    return images

def load_generators(train_path, validation_path, batch_size=32, preprocess=True):
    '''
    This function creates `keras.preprocessing.image.ImageDataGenerator` objects 
    for images located in `train_path` and `validation_path`. 
    
    The batch size can be specified.  
    Whether or not images should be preprocesses using `keras.applications.vgg16.preprocess_input`
    ImageNet mean subtraction is applied to both ImageGenerators.

    This function is based off code from
    https://www.pyimagesearch.com/2019/06/03/fine-tuning-with-keras-and-deep-learning/

    Inputs:
    -------------------------
    train_path: str, path to directory for training images
    validation_path: str, path to directory for validation images
    batch_size: int (default = 32)
        * the batch size to be used in the validation generator
    preprocess: boolean (default = True)
        * whether or not images should be preprocesses using `keras.applications.vgg16.preprocess_input`
    
    Returns:
    -------------------------
    train_generator, validation_generator
    
    '''
    # Initialize the training data augmentation object
    if preprocess:
        train_datagen = ImageDataGenerator(
            rotation_range=30,
            zoom_range=0.15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            horizontal_flip=True,
            fill_mode="nearest",
            preprocessing_function=preprocess_input)
    else:
        train_datagen = ImageDataGenerator(
            rotation_range=30,
            zoom_range=0.15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            horizontal_flip=True,
            fill_mode="nearest")

    # Initialize the validation/testing data augmentation object
    if preprocess:
        validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    else:
        validation_datagen = ImageDataGenerator()

    # Define the ImageNet mean subtraction (in RGB order) and set the
    # the mean subtraction value for each of the data augmentation objects
    mean = np.array([123.68, 116.779, 103.939], dtype="float32")
    train_datagen.mean = mean
    validation_datagen.mean = mean

    # Initialize the training generator
    train_generator = train_datagen.flow_from_directory(
        train_path,
        class_mode="categorical",
        target_size=(256, 256),
        color_mode="rgb",
        shuffle=True,
        batch_size=batch_size)

    # Initialize the validation generator
    validation_generator = validation_datagen.flow_from_directory(
        validation_path,
        class_mode="categorical",
        target_size=(256, 256),
        color_mode="rgb",
        shuffle=False,
        batch_size=batch_size)
    
    return train_generator, validation_generator

def plot_results(history):
    '''
    This function takes the history of a keras model and plots it.
    The loss (training and validation) and accuracy (training and validation) is plotted.

    Inputs:
    -------------------------
    history: tf.keras.callbacks.History or dict
        * dictionary comes from tf.keras.callbacks.History.history
        * dict must contain the keys ['loss', 'val_loss', 'accuracy', 'val_accuracy']
    

    Returns:
    -------------------------
    fig: matplotlib.pyplot.figure
    
    '''
    try: 
        # Get history dictionary if history is a tf.keras.callbacks.History object
        history = H.history
    except: 
        pass

    fig, ax = plt.subplots(1, 2, figsize=(15, 5), sharex=True)
    ax[0].plot(history['loss'], label='training')
    ax[0].plot(history['val_loss'], label='validation')
    ax[1].plot(history['accuracy'], label='training')
    ax[1].plot(history['val_accuracy'], label='validation')
    ax[0].set_yscale("log")
    ax[0].set_ylabel("Loss")
    ax[1].set_ylabel("Accuracy")
    ax[0].set_xlabel("Epochs")
    ax[1].set_xlabel("Epochs")
    ax[0].legend()
    ax[1].legend()
    fig.show();

    return fig


def prepare_inputs(images, preprocess=True):
    '''
    This function takes a list of images and prepares them to be passed as 
    inputs to keras.applications.VGG16.

    Inputs:
    -------------------------
    images: list of cv2 images / 2D arrays / 3D arrays
    preprocess: boolean (default = True)
        * whether or not images should be preprocesses using `keras.applications.vgg16.preprocess_input`

    Returns:
    -------------------------
    tf_imgs: list of cv2 images / 2D arrays / 3D arrays
    '''
    tf_imgs = np.array([cv2.resize(img, (256, 256)) for img in images])
    if preprocess:
         tf_imgs = preprocess_input(tf_imgs)

    return tf_imgs


def process_results(output, thresh=0.5):
    '''
    This function takes the output from a kears model and processes them.

    The first element in each output (the sip score) is saved in scores.
    This score is then thesholded to 0 or 1 according to the `thresh` 
        argument and saved in preds.
    This prediction is then converted to 'sip!' or 'skip :(' using the 
        class_labels dictionary. These are saved in labels.

    Inputs:
    -------------------------
    output: <tf.Tensor: shape=(n, 2), dtype=float32>
        * the outputs from a keras model (with 2 output dimensions)
    thresh: float, (default = 0.5)
        * the threshold to apply to the first element in each output row
        * values will be thresholded to 0 or 1 according to this parameter

    Returns:
    -------------------------
    scores: list of floats (coressponding to the sip score)
    preds: list of ints (thesholded scores)
    labels: list of str (class labels of preds)
    '''

    # Extract numpy array from tensor
    try:
        output = output.numpy()
    except:
        pass

    # Extract sip scores
    scores = output[:, 0]

    # Apply threshold to sip scores
    preds = (1 - scores > 1 - thresh).astype(int)

    # Convert predictions to class labels
    labels = [class_labels[p] for p in preds]

    return scores, preds, labels


def plot_roc(y_true, y_score, thresh=None, show_fig=True, label=None):
    '''
    This function plots the ROC curve corresponding to a set of y_true and
    y_pred values. A certain threshold can be indicated on the ROC curve by
    specifying a `thresh`.


    Inputs:
    -------------------------
    y_true: list of ints
        * true labels

    y_score: list of floats
        * predicted scores from keras model

    thresh: float (default = None)
        * threshold value to mark on the ROC curve
        * if None, no mark is made

    show_fig: bool, (default = False)
        * whether or not to call fig.show() in-function
        * if False, the ROC curve will be plotted on the active figure, allowing
          for multiple ROC curves to be plotted on the same figure
    
    label: str, (default = None)
        * label for the ROC curve
        * If None, the label will only contain the AUC of the ROC curve
        * If a value, the label will preced the AUC report

    Returns:
    -------------------------
    None
    '''
    # Compute fpr, tpr, thresholds and roc auc
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    # The y_score values may need to be adjusted so that the AUC > 0.5.
    if tpr[-len(tpr) // 2] < 0.5:
        y_score = 1 - y_score                               # Sip scores must correspond to 0 = sip, 1 = skip
        fpr, tpr, thresholds = roc_curve(y_true, y_score)   # Re-calculate fpr and tpr

    # Calculate ROC
    roc_auc = auc(y_true, y_score)

    # Create te label for the ROC curve
    if label:
        label = '{:}, ROC curve (area = {:0.3f})'.format(label, roc_auc)
    else:
        label = 'ROC curve (area = {:0.3f})'.format(roc_auc)

    # Plot ROC curve
    plt.plot(fpr, tpr, label=label)
    plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate or (1 - Specifity)')
    plt.ylabel('True Positive Rate or (Sensitivity)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    # Mark on the plot what the true/false positive rates are for a certain threshold
    if thresh is not None:
        idx = (np.abs(thresholds - thresh)).argmin()
        plt.plot(fpr[idx], tpr[idx], 'ko')
        plt.vlines(fpr[idx], 0, 1, linestyle='-.', lw=0.5)
        plt.hlines(tpr[idx], 0, 1, linestyle='-.', lw=0.5)
        plt.text(0.8, 0.15, "Threshold = {:.2f}".format(thresholds[idx]), horizontalalignment="center", bbox={"facecolor":"white", "linewidth":0.25})
    
    # If show_fig = False, ROC curve will be plotted on active fig
    if show_fig:
        plt.show();


def get_tpr_fpr(y_true, y_score, thresh=0.5):
    '''
    This function finds the true- and false-positive rates of predictions
    at a specified threshold.

    Inputs:
    -------------------------
    y_true: list of ints
        * true labels

    y_score: list of floats
        * predicted scores from keras model

    thresh: float (default = None)
        * threshold value to mark on the ROC curve
        * if None, no mark is made

    Returns:
    -------------------------
    tpr: float
        * true positive rate of the predictions at the specififed threshold
    fpr: float
        * false positive rate of the predictions at the specififed threshold
    '''
    # Compute fpr, tpr, thresholds and roc auc
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    # The y_score values may need to be adjusted so that the AUC > 0.5.
    if tpr[-len(tpr) // 2] < 0.5:
        y_score = 1 - y_score                               # Sip scores must correspond to 0 = sip, 1 = skip
        fpr, tpr, thresholds = roc_curve(y_true, y_score)   # Re-calculate fpr and tpr

    # Find the closest threshold
    idx = (np.abs(thresholds - thresh)).argmin()

    return tpr[idx], fpr[idx]


def plot_predictions(imgs, scores, y_true=None, num_to_plot=9):
    '''
    This function plots num_to_plot images and their predictions in a grid-like
    pattern, all on the same figure.

    This function makes it useful to visualize the difference in predictions
    that each model makes.


    Inputs:
    -------------------------
    imgs: list of cv2 images / 2D arrays / 3D arrays --OR-- a single cv2 image
    scores: dict
        * sip scores corresponding to imgs
        * scores = {
            'google1': [0.56, 0.40, 0.97, 0.22, ...],
            'google2': [0.50, 0.44, 0.91, 0.31, ...],
            ...
        }
    y_true: list of ints (default = None)
        * the true classifications of each image
        * 0 = sip, 1 = skip
        * if none are given, the true value will not be displayed in the axis titles
    num_to_plot: int (default = 9)
        * number of images to plot in the figure

    Returns:
    -------------------------
    None
    '''
    # Handle single-image inputs
    if type(imgs) != list:
        imgs = [imgs]

    num_to_plot = min(num_to_plot, len(imgs))

    # Calculate the number of rows needed for the plot
    n_cols = min(3, len(imgs), num_to_plot)
    n_rows = max(1, num_to_plot // n_cols)
    
    # Randomly select num_to_plot images
    idx = np.random.choice(np.arange(len(imgs)), num_to_plot)

    # Loop over the selected images
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(25, 7*n_rows))
    to_loop = [ax] if num_to_plot == 1 else ax.flatten()
    for i, axes in enumerate(to_loop):
        # Get the image, its prediction, and its true label
        img = imgs[idx[i]]
        y_preds = [scores[m][idx[i]] for m in model_names]

        # Create the text box with the results
        text = 'Predictions:\n----------------------\n{0}: {5:8.1%}\n{1}: {6:8.1%}\n{2}: {7:8.1%}\n{3}: {8:8.1%}\n{4}: {9:8.1%}'.format(*model_names, *y_preds)

        # Plot the image and format the axis
        axes.imshow(img)
        axes.set_yticks([])
        axes.set_xticks([])
        axes.text(270, 150, s=text, fontsize='xx-large')

        # Display true labels (if known)
        if y_true is not None:
            true_label = y_true[idx[i]]
            axes.set_title("True label: {} {:.0%}".format(class_labels[true_label], 1-true_label), fontsize='xx-large')

    # Format the figure
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.8, hspace=0)
    fig.show();


def get_purity(preds):
    '''
    This function returns the purity of a set of predictions.

    Purity is defined as the number of observations with the majority prediction
    divided by the total number of observations.
    
    For example, if the input contains 2 observations with label 0 and 3 
    observations with label 1, the purity would be max(2, 3) / (2 + 3) = 0.6.

    This is used instead of more conventional purity metrics because the sample
    size is relatively small (5).

    Inputs:
    -------------------------
    preds: list of binary values
        * the set of binary predictions
        * ex. [0, 1, 0, 0, 1]

    Returns:
    -------------------------
    purity: float
        * ex. 0.6
    '''
    n0 = sum(preds) 
    n1 = len(preds) - n0
    return max(n0, n1) / len(preds)


def get_variability_stats(scores, preds):
    '''
    This function returns variability statistics about a set of scores and 
    predictions. The supposted statistics include:
        * mean score
        * std. dev. of score
        * range of scores
        * purity of predictions

    Purity is defined as the number of observations with the majority prediction
    divided by the total number of observations. For example, if the input 
    contains 2 observations with label 0 and 3 observations with label 1, the 
    purity would be max(2, 3) / (2 + 3) = 0.6. This is used instead of more 
    conventional purity metrics because the sample size is relatively small (5).

    Inputs:
    -------------------------
    scores: list of floats
        * the set of sip scores
        * ex. [0, 1, 0, 0, 1]
    
    preds: list of binary values
        * the set of binary predictions
        * ex. [0, 1, 0, 0, 1]

    Returns:
    -------------------------
    mean: float
    std: float
    rng: float
    purity: float
    '''
    mean = scores.mean()
    std = scores.std()
    rng = scores.max() - scores.min()
    purity = get_purity(preds)

    return mean, std, rng, purity


def get_name_variability(name, model, thresh=0.5, img_dir="..\friends\face_only", preprocess=True, generator=None, verbose=False):
    '''
    This function returns variability statistics for all images containing a
    specific name. The supposted statistics include:
        * mean score
        * std. dev. of score
        * range of scores
        * purity of predictions

    Purity is defined as the number of observations with the majority prediction
    divided by the total number of observations. For example, if the input 
    contains 2 observations with label 0 and 3 observations with label 1, the 
    purity would be max(2, 3) / (2 + 3) = 0.6. This is used instead of more 
    conventional purity metrics because the sample size is relatively small (5).

    Inputs:
    -------------------------
    name: str, the name of the person to get images of
    model: keras model, the model to be used for predicting
    thresh: float (default = 0.5)
        * the threshold level to be used on the predicted sip score to be
          labeled as such 
    img_dir: str, (default = "..\friends\face_only")
        * the path to the directory containing the images
        * ignored if `generator` is not None
    preprocess: Boolean, (default = True)
        * whether or not to preprocess the images before predicting
        * ignored if `generator` is not None
    generator: keras.preprocessing.image.ImageDataGenerator (default = None)
        * the image generator containing the images
        * if None, the generator will be compiled by finding all images of 
          `name` in `img_dir`
    verbose: Boolean, (default = False)
        * whether or not to print out functino progress

    Returns:
    -------------------------
    mean: float
    std: float
    rng: float
    purity: float
    '''
    if generator is None:
        # Load images of the specified person
        if verbose: print("Using name: {}\n".format(name))
        img_paths = glob.glob(img_dir + '/' + name + '*.jpg')
        named_images = load_image_set(img_paths, face_only=False, verbose=verbose)
        if verbose: print("Done.")

        # Preprocess images
        generator = prepare_inputs(named_images, preprocess=preprocess)

    # Predict image labels (0 = sip, 1 = skip)
    output = model(generator)
    scores, preds, labels = process_results(output, thresh)

    # Get variability statistics
    mean, std, rng, purity = get_variability_stats(scores, preds)

    return mean, std, rng, purity


def visualize_heatmap(imgs, heatmaps, scores, preds, true_labels=None, savepath=None):
    '''
    This function plots images and their LRP heatmaps, all on the same figure.

    This function makes it useful to visualize which pixels in an image had
    the largest contribution to the model's prediction.


    Inputs:
    -------------------------
    imgs: list of cv2 images / 2D arrays / 3D arrays --OR-- a single cv2 image
    scores: list of floats --OR-- a single float
        * sip scores corresponding to imgs
        * scores = [0.56, 0.40, 0.97, 0.22, ...]
    preds: list of ints --OR-- a single int
        * class labels corresponding to imgs
        * 0 = sip, 1 = skip
        * scores = [1, 0, 1, 0, ...]
    y_true: list of ints (default = None)
        * the true classifications of each image
        * 0 = sip, 1 = skip
        * if none are given, the true value will not be displayed in the axis titles

    Returns:
    -------------------------
    None
    '''
    if type(imgs) != list:
        imgs = [imgs]
        heatmaps = [heatmaps]
        scores = [scores]
        preds = [preds]
    
    # Calculate the number of rows needed for the plot
    n_cols = 2
    n_rows = 2 * len(imgs) // n_cols
    
    # Loop over the selected images
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(8, 5*n_rows))
    for i, axes in enumerate(ax.flatten()):
        if not i % 2:
            axes.axis('off')
            axes.imshow(imgs[i // 2])

            if true_labels is not None:
                axes.set_title("True label: {} {:.0%}".format(class_labels[true_labels[i // 2]], 1-true_labels[i // 2]), fontsize='xx-large')

        else:
            axes.axis('off')
            axes.imshow(heatmaps[i // 2], interpolation='bilinear')
            axes.set_title("Prediction: {} {:.2%}".format(class_labels[preds[i // 2]], scores[i // 2]), fontsize='xx-large')

    if savepath is not None:
        fig.savefig(savepath)
    
    # Format the figure
    fig.tight_layout()
    fig.subplots_adjust(wspace=0)
    fig.show();