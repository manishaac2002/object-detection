import cv2
# shoutld add the confidence level

#importing image and processing image file
# img = cv2.imread('dhoni.jpeg')
cap =cv2.VideoCapture(0)# 0's and 1's for camera
cap.set(3,640)
cap.set(4,480)

#setting up the coco names 
classNames =[]
#config file of coco names
classFile ='coco.names'
with open(classFile,'rt') as f:
    classNames =f.read().rstrip('\n').rsplit('\n')
   
#config file 
configPath ='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath ='frozen_inference_graph.pb'

# getting config file
net = cv2.dnn_DetectionModel(weightsPath,configPath)

# setting bounding box
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)

while True:
    success,img =cap.read()
    classIds ,conf ,bbox =net.detect(img,confThreshold=0.5)
    print(classIds,bbox)

    if len(classIds) !=0:
        #parameter for bounding box
        for classIds,conf,bbox in zip(classIds.flatten(),conf.flatten(),bbox):
            cv2.rectangle(img,bbox,color=(0,255,0),thickness=2)
            cv2.putText(img,classNames[classIds-1],(bbox[0]+10,bbox[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)


#output image
    cv2.imshow("Output",img)
    cv2.waitKey(1)# waitKey for pic(0's) and live camera(1's)