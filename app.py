#Import necessary libraries
from flask import Flask, flash, render_template, Response
from flask import redirect, url_for, request
import cv2
from keras.models import load_model
import pickle
import numpy as np
import json
import winsound
import time
from pygame import mixer

#Initialize the Flask app
app = Flask(__name__,template_folder='templates',static_folder='static')
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'


BEST_MODEL = "static/distracted-24-0.99.hdf5"
(H, W) = (None, None)

model = load_model(BEST_MODEL)

with open("static/labels.pkl","rb") as handle:
    labels_id = pickle.load(handle)

print(labels_id)

#predict function
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


    with open('class_name_map.json','w') as secret_input:
        json.dump(class_name,secret_input,indent=4,sort_keys=True)

    with open('class_name_map.json') as secret_input:
        info = json.load(secret_input)
        label = info[id_labels[ypred_class]]
        print(label)
    
    return label





camera = cv2.VideoCapture(0)





def gen_frames(camera):  
    while True:
        success, frame = camera.read()  # read the camera frame
        output=frame.copy()
        (H, W) = frame.shape[:2]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (128, 128))
            
        # frame -= mean
        frame = np.expand_dims(frame,axis=0).astype('float32')/255 - 0.5
            

        # make predictions on the frame and then update the predictions
        # queue
        label = predict_result(frame)
        print(label)

        #alarm
        
        # duration = 5000  # milliseconds
        # freq = 440  # Hz

        # if label is not "safe driving":
        #     time.sleep(1)
        #     winsound.Beep(freq, duration)

        mixer.init()
        sound = mixer.Sound('alarm.wav')

        if label != "SAFE_DRIVING":
            sound.play()






        #frame = np.squeeze(frame,axis=0).astype('float32')*400 + 0.5


        text = "activity: {}".format(label)
        cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,
        1.25, (0, 255, 0), 5)
            
        
            



        ret, buffer = cv2.imencode('.jpg', output)
        output = buffer.tobytes()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + output + b'\r\n')  # concat frame one by one and show result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/cam')
def cam():
    return render_template('cam.html')

@app.route('/example')
def example():
    return render_template('example.html')

# @app.route('/example')
# def example():
#     return redirect('/static/output_video.mp4')

@app.route('/video_feed')
def video_feed():
    global camera
    return Response(gen_frames(camera), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/dash')
def dash():
    return render_template('dash.html')

# @app.route('/login')
# def login():
#     return render_template('login.html')

@app.route('/login',methods = ['POST', 'GET'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['eid'] != 'A101' or request.form['pass'] != 'e123':
            error = 'Invalid credentials'
        else:
            flash('You were successfully logged in')
            return redirect(url_for('login'))
    return render_template('login.html', error=error)
      
      

    

    #return redirect(url_for('success',name = user))

if __name__ == "__main__":
    app.run(debug=True)