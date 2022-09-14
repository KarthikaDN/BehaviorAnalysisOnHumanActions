from flask import Flask,render_template,request,url_for,redirect,flash,session,make_response    
import cv2
import mediapipe as mp
import numpy as np
import os
from centroidtracker import CentroidTracker
import datetime
import imutils
import os
import time
import urllib.request
from werkzeug.utils import secure_filename



mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
UPLOAD_FOLDER = 'static/uploads/'

app=Flask(__name__) 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024


@app.route("/",methods=["POST","GET"])
def frontpage():
    return render_template("frontpage.html")

@app.route("/enter", methods=["POST","GET"])
def uploadVideo():
    return render_template('upload.html')

@app.route("/upload",methods=['POST','GET'])
def  login():
    return render_template("upload.html")

@app.route("/uploadForCount",methods=["POST","GET"])
def upload_video_for_count():
    
    file = request.files['file']
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    print('')

    protopath = "MobileNetSSD_deploy.prototxt"
    modelpath = "MobileNetSSD_deploy.caffemodel"
    detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)
    # detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
    # detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
            "sofa", "train", "tvmonitor"]

    tracker = CentroidTracker(maxDisappeared=80, maxDistance=90)

    def non_max_suppression_fast(boxes, overlapThresh):
        try:
            if len(boxes) == 0:
                return []

            if boxes.dtype.kind == "i":
                boxes = boxes.astype("float")

            pick = []

            x1 = boxes[:, 0]
            y1 = boxes[:, 1]
            x2 = boxes[:, 2]
            y2 = boxes[:, 3]

            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            idxs = np.argsort(y2)

            while len(idxs) > 0:
                last = len(idxs) - 1
                i = idxs[last]
                pick.append(i)

                xx1 = np.maximum(x1[i], x1[idxs[:last]])
                yy1 = np.maximum(y1[i], y1[idxs[:last]])
                xx2 = np.minimum(x2[i], x2[idxs[:last]])
                yy2 = np.minimum(y2[i], y2[idxs[:last]])

                w = np.maximum(0, xx2 - xx1 + 1)
                h = np.maximum(0, yy2 - yy1 + 1)

                overlap = (w * h) / area[idxs[:last]]

                idxs = np.delete(idxs, np.concatenate(([last],
                                                    np.where(overlap > overlapThresh)[0])))

            return boxes[pick].astype("int")
        except Exception as e:
            print("Exception occurred in non_max_suppression : {}".format(e))

    
    cap = cv2.VideoCapture("./static/uploads/"+filename)

    fps_start_time = datetime.datetime.now()
    fps = 0
    total_frames = 0
    lpc_count = 0
    opc_count = 0
    object_id_list = []
    accuracy=[]
    while True:
        
        ret, frame = cap.read()
        if frame is None:
            break
        frame = imutils.resize(frame, width=600)
        total_frames = total_frames + 1

        (H, W) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)

        detector.setInput(blob)
        person_detections = detector.forward()
        rects = []
        for i in np.arange(0, person_detections.shape[2]):
            confidence = person_detections[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(person_detections[0, 0, i, 1])

                if CLASSES[idx] != "person":
                    continue

                person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = person_box.astype("int")
                rects.append(person_box)

        boundingboxes = np.array(rects)
        boundingboxes = boundingboxes.astype(int)
        rects = non_max_suppression_fast(boundingboxes, 0.3)

        objects = tracker.update(rects)
        for (objectId, bbox) in objects.items():
            x1, y1, x2, y2 = bbox
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            text = "ID: {}".format(objectId)
            cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

            if objectId not in object_id_list:
                object_id_list.append(objectId)

        fps_end_time = datetime.datetime.now()
        time_diff = fps_end_time - fps_start_time
        if time_diff.seconds == 0:
            fps = 0.0
        else:
            fps = (total_frames / time_diff.seconds)

        fps_text = "FPS: {:.2f}".format(fps)

        cv2.putText(frame, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

        lpc_count = len(objects)
        opc_count = len(object_id_list)
        
        if lpc_count>1:         
            accuracy.append('yes')
            # print('ABNORMAL:More than one person detected!')
            cv2.putText(frame, 'ABNORMAL:More than one person detected!', (5, 90), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
        else:
            accuracy.append('no')
        lpc_txt = "Live person: {}".format(lpc_count)
        opc_txt = "OPC: {}".format(opc_count)

        cv2.putText(frame, lpc_txt, (5, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
    #         cv2.putText(frame, opc_txt, (5, 90), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
        

        cv2.imshow("Application", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(accuracy)
    return redirect('upload')



@app.route("/uploadForhelmet",methods=["POST","GET"])
def upload_video_for_helmet():
    file = request.files['file']
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    print('')

    net = cv2.dnn.readNet("yolov3-obj_2400.weights", "yolov3-obj.cfg")
    classes = []
    with open("obj.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    color = (255, 0, 0)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    camera = cv2.VideoCapture("./static/uploads/"+filename)
    HelmetAccuracyList=[]
    while True:
        
        _,img = camera.read()
        if img is None:
            break
        height, width, channels = img.shape

        blob = cv2.dnn.blobFromImage(img, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        font = cv2.FONT_HERSHEY_PLAIN
        if len(boxes)>0:
            HelmetAccuracyList.append('Detected')
            cv2.putText(img, 'Helmet Detected!!!', (50, 90), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
        else:
            HelmetAccuracyList.append('notDetected')
            
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
    #             color = colors[i]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
        cv2.namedWindow('video',cv2.WINDOW_NORMAL)
        # ims = cv2.resize(img,(1500,1500))
        cv2.imshow('video', img)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()
    print(HelmetAccuracyList)
    return redirect('upload')

@app.route("/uploadForPose",methods=['POST','GET'])
def upload_video_for_pose():
    file = request.files['file']
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    
    def calculate_angle(a,b,c):
        a = np.array(a) # First
        b = np.array(b) # Mid
        c = np.array(c) # End
    
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
    
        if angle >180.0:
            angle = 360-angle
        
        return angle
    cap=cv2.VideoCapture('./static/uploads/'+filename)
    # cap = cv2.VideoCapture(0)
    flagBendR=0
    flagBendL=0
    flagRiseR=0
    flagRiseL=0
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark
                
                shoulderR = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                hipR=  [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                kneeR = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                
                shoulderL = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                hipL=  [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                kneeL = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                
                elbowR = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                elbowL = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                
                
                angle1 = calculate_angle(shoulderR,hipR,kneeR)
                if angle1<110:

                    cv2.putText(image, 'ABNORMAL DETECTED right hip angle', (50, 90), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)

                    #return "ABNORMAL DETECTED right hip angle->"
                    # print("ABNORMAL DETECTED right hip angle->",angle1,"------------------",dt_string)
                    
                angle2 = calculate_angle(shoulderL,hipL,kneeL)
                if angle2<110:
                    cv2.putText(image, 'ABNORMAL DETECTED left hip angle', (50, 190), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
                    #return "ABNORMAL DETECTED left hip angle->"
                    # print("ABNORMAL DETECTED right hip angle->",angle2,"------------------",dt_string)
                
                angle3 = calculate_angle(hipR,shoulderR,elbowR)
                if angle3>120:
                    cv2.putText(image, 'ABNORMAL DETECTED right elbow angle', (50, 290), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
                    #return "ABNORMAL DETECTED right elbow angle->"
                    # print("ABNORMAL DETECTED right right elbow->",angle3,"------------------",dt_string)
                    
                angle4 = calculate_angle(hipL,shoulderL,elbowL)
                if angle4>120:
                    cv2.putText(image, 'ABNORMAL DETECTED left elbow angle', (50, 390), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
                    #return "ABNORMAL DETECTED left elbow angle->"
                    # print("ABNORMAL DETECTED right right elbow->",angle4,"------------------",dt_string)
                
            except:
                pass

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=3), 
                                mp_drawing.DrawingSpec(color=(0,255,255), thickness=2, circle_radius=2) 
                                 )      
        
            cv2.namedWindow('video',cv2.WINDOW_NORMAL)
        
            # ims = cv2.resize(image,(1500,1500))
        
            cv2.imshow('video', image)
            retval, buffer = cv2.imencode('.png', image)
            response = make_response(buffer.tobytes())
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    return redirect('upload')
    
if __name__ == "__main__":
    app.run(debug=True)