import os
import json
from keras.models import load_model
import pandas as pd
import pickle
import numpy as np
import shutil
from collections import deque
import numpy as np
import argparse
import pickle
import cv2

from keras.preprocessing import image                                              

BEST_MODEL = "static/distracted-07-0.98.hdf5"
model = load_model(BEST_MODEL)

with open("static/labels.pkl","rb") as handle:
    labels_id = pickle.load(handle)

def predict_result(image_tensor):

    ypred_test = model.predict(image_tensor,verbose=1)
    ypred_class = np.argmax(ypred_test,axis=1)
    print(ypred_class)

    id_labels = dict()
    for class_name,idx in labels_id.items():
        id_labels[idx] = class_name
    ypred_class = int(ypred_class)
    print(id_labels[ypred_class])


    #to create a human readable and understandable class_name 
    class_name = dict()
    class_name["c0"] = "SAFE_DRIVING"
    class_name["c1"] = "TEXTING_RIGHT"
    class_name["c2"] = "TALKING_PHONE_RIGHT"
    class_name["c3"] = "TEXTING_LEFT"
    class_name["c4"] = "TALKING_PHONE_LEFT"
    class_name["c5"] = "OPERATING_RADIO"
    class_name["c6"] = "DRINKING"
    class_name["c7"] = "REACHING_BEHIND"
    class_name["c8"] = "HAIR_AND_MAKEUP"
    class_name["c9"] = "TALKING_TO_PASSENGER"


    with open("class_name_map.json",'w') as secret_input:
        json.dump(class_name,secret_input,indent=4,sort_keys=True)

    with open("class_name_map.json") as secret_input:
        info = json.load(secret_input)
        label = info[id_labels[ypred_class]]
        print(label)
    
    return label


INPUT_VIDEO_FILE = "input_video.mp4"
OUTPUT_VIDEO_FILE = "output_video.mp4"
# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(INPUT_VIDEO_FILE)
writer = None
(W, H) = (None, None)
# loop over frames from the video file stream
while True:
	# read the next frame from the file
	(grabbed, frame) = vs.read()
	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break
	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]
	# clone the output frame, then convert it from BGR to RGB
	# ordering, resize the frame to a fixed 224x224, and then
	# perform mean subtraction
	output = frame.copy()
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	frame = cv2.resize(frame, (128, 128))
	# frame -= mean
	frame = np.expand_dims(frame,axis=0).astype('float32')/255 - 0.5

	# make predictions on the frame and then update the predictions
	# queue
	label = predict_result(frame)
	# preds = model.predict(np.expand_dims(frame, axis=0))[0]
	# Q.append(preds)
	
	# perform prediction averaging over the current history of
	# previous predictions
	# results = np.array(Q).mean(axis=0)
	# i = np.argmax(results)
	# label = lb.classes_[i]

		# draw the activity on the output frame
	text = "activity: {}".format(label)
	cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,
		1.25, (0, 255, 0), 5)
	# check if the video writer is None
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(OUTPUT_VIDEO_FILE, fourcc, 30,
			(W, H), True)
	# write the output frame to disk
	writer.write(output)
	# show the output image
	cv2.imshow("Output", output)
	key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()

