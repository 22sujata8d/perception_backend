from __future__ import print_function
import requests
import json
import cv2

# prepare the address for the post request
addr = 'http://localhost:5000'
test_url = addr + '/api/test'

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}

# read the image 
img = cv2.imread('test.jpeg')

# encode image as jpeg
_, img_encoded = cv2.imencode('.jpeg', img)
print(type(img_encoded))

# send http request with image and receive response
response = requests.post(test_url, data=img_encoded.tostring(), headers=headers)

# decode response
decode = json.loads(response.text)

# print the info obtained from the response
print(decode)
#print(decode, " ", type(decode), " ", decode['message'])
#print(decode['probabilities'])
