from flask import Flask, request, make_response
from flask_cors import CORS
import re
from base64 import b64decode
from PIL import Image
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return 'You hit the HOME route!'

@app.route('/api/calibrate', methods=["POST"])
def calibrate():
    # calculates focal length of user's camera
    known_distance = abs(float(request.form['distance']))+0.001
    known_width = abs(float(request.form['width']))+0.001
    print("d: {}, w: {}".format(known_distance, known_width))
    img = read_in_image_file(request.form['file'])

    pd_pixels = find_eyes(img)

    # use PD to find focal length of your camera
    focal_length = (pd_pixels * known_distance) / known_width
    print("focal length: {}".format(focal_length))

    return {
        "message": "Calibrated!",
        "focal_length": focal_length,
        "distance": known_distance-0.001,
        "width": known_width-0.001
    }

@app.route('/api/estimate', methods=["POST"])
def estimate():
    known_distance = abs(float(request.form['distance']))+0.001
    known_width = abs(float(request.form['width']))+0.001
    focal_length = abs(float(request.form['focalLength']))+0.001

    img = read_in_image_file(request.form['file'])
    cm = distance_to_camera(img, known_width, focal_length)
    diopters = 100 / (cm+0.001)
    return {
        "message": "Estimated!",
        "cm": cm,
        "diopters": diopters,
        "focal_length": focal_length-0.001,
        "distance": known_distance-0.001,
        "width": known_width-0.001
    }

# find eyes in image and return PD in pixels
def find_eyes(img):
    # Load the cascades
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        big_x = x
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

    eyes = eye_cascade.detectMultiScale(gray)
#    print(eyes)
    centers = []
    for (ex,ey,ew,eh) in eyes:
        centers.append(ex+(ew/2))
        cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    
    # calculate PD in pixels from image
#    print(centers)
#    print(big_x)
    return centers[1] - centers[0]

# determine estimate distance user is from their camera
def distance_to_camera(img, known_width, focal_length):
    pd_px = find_eyes(img)
    distance = (known_width * focal_length) // pd_px

    # compute and return the distance from the marker to the camera
    return distance

def read_in_image_file(request_file):
    image_data = re.sub('^data:image/.+;base64,', '', request_file)
    #read image file string data
    filestr = b64decode(image_data)
    #convert string data to numpy array
    npimg = np.fromstring(filestr, np.uint8)
    # convert numpy array to image
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    return img
