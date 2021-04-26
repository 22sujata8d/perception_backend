import sys
sys.path.append("/home/sujata/Backend")

import cv2
from model.predict import *
from flask import Flask, request, Response, jsonify
import jsonpickle
import json
import numpy as np
#import PIL.Image as Image, ImageOps
from PIL import Image, ImageOps

# Initialize the Flask application
app = Flask(__name__)

# preprocess the input image


def preprocess_image(img):
    # resize image to (28, 28) for model
    height = 28
    width = 28
    dimension = (width, height)
    preprocessed_img = cv2.resize(img, dimension)

    return preprocessed_img

# route http posts to this method
@app.route('/api/test', methods=['POST'])
def test():
    r = request
    content = r.get_json(silent=True)
    img = content["image"]
    print(type(img))

    img_arr = np.asarray(img, dtype=np.uint8)
    print(type(img_arr))

    # convert string of image data to uint8
    #print(type(r))
    #print(r)
    #data = json.loads(r)
    #s = json.dumps(data, indent=4, sort_keys=True)
    #print(s)
    #print(s["image"])
    #nparr = np.frombuffer(r.data, np.uint8)
    
    #print(r)

    # decode image

    img = cv2.imdecode(img_arr, cv2.IMREAD_GRAYSCALE)

    resized_img = preprocess_image(img)

    #save_img = Image.fromarray(np.uint8(cm.gist_earth(resized_img)*255))

    #save_img = save_img.convert('L')

    #save_img.save("asitis.jpeg")

    #revert_img = cv2.bitwise_not(save_img)

    revert_img = ImageOps.invert(save_img)

    #revert_img.save("revert.jpeg")

    #img = Image.open(io.BytesIO(r.data))

    print("Reading image .... ")

    # get probabitities for the output labels 0 to 9
    prob = get_probabilities(revert_img)

    print(prob)
    #print(type(prob))

    # convert tensor into list of decimals
    prob_list = prob.squeeze().tolist()
    # convert list of decimals into list of strings
    prob_string = [str(n) for n in prob_list]

    print(prob_string)

    # build a response dict to send back to client
    response = {'message': 'image received. size={}x{}'.format(resized_img.shape[1], resized_img.shape[0]),
                'probabilities': prob_string}

    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")


# start flask app
app.run(host="0.0.0.0", port=5000)
