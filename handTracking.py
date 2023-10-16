import cv2
import mediapipe as mp
import time
import csv

#輸出格式-> 手指節點

capturedFrame = cv2.VideoCapture(0)         ##內部攝影機的編號為0
fps = 20
capturedFrame.set(cv2.CAP_PROP_FPS, fps)     ##設定攝影機的FPS


isRecording:bool = False
recordedDataList:list = []
sortedData = []

handsModel = mp.solutions.hands             ##手部檢測的解決方案??
hands=handsModel.Hands(min_tracking_confidence=0.2)        ##手部檢測模型的實例
mpDraw = mp.solutions.drawing_utils         ##繪製不同檢測結果的模組
markStyle=mpDraw.DrawingSpec(color=(0,255,0),thickness=5)
connectionStyle=mpDraw.DrawingSpec(color=(255,255,255),thickness=2)
previousTime = 0

def getFpsAndUpdateTime(previousTime):
    fps= 1/(time.time()-previousTime)
    previousTime = time.time()   
    cv2.putText(image, f"fps : {int(fps)}",(30,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),1)
    return previousTime

def SortData(originalData):
    leftHandData=[]
    rightHandData=[]
    regData=[]
    outputData=[]

    for item in originalData:
        if item['leftOrRight'] == 'Left':
            leftHandData.append(item)
        elif item['leftOrRight'] == 'Right':
            rightHandData.append(item)
    for i in rightHandData:
        i['node']= -i['node']
    

    regData.append(leftHandData)
    regData.append(rightHandData)
    outputData=[]
    for i in regData:
        data = [[d['node'], d['deltaX'], d['deltaY']] for d in i]
        outputData.extend(data)
    
    return outputData


def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        StartRecord()
     

def StartRecord():
    global isRecording
    global recordStartTime
    isRecording = True       
    recordStartTime = time.time()

def DrawHandMarks():
    global result
    global image
    if result.multi_hand_landmarks:         #劃出點點
        for handLandmarks in result.multi_hand_landmarks:
            # 在影像中畫出手部關鍵點
            for point in handLandmarks.landmark:
                mpDraw.draw_landmarks(image, handLandmarks, handsModel.HAND_CONNECTIONS, markStyle,connectionStyle)


def RecordData(i,x,y,isRightHand):
    global isRecording
    deltaX = 0
    deltaY = 0
    if recordedDataList:
        deltaX = x-recordedDataList[-1]['x']
        deltaY = y-recordedDataList[-1]['y']
    if isRightHand:
        leftOrRight:str = 'Right'
    else:
        leftOrRight:str = 'Left'
    if (time.time() - recordStartTime < 1 )and(time.time()):
        recordedDatum:dict = {'node':i,'x':x,'y':y,'deltaX':deltaX,'deltaY':deltaY,'leftOrRight':leftOrRight}
        recordedDataList.append(recordedDatum)
    else:
        isRecording = False
        print('recorded once')


def ShowHandsNode():
    if result.multi_hand_landmarks:
        for handMarks in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(image, handMarks, handsModel.HAND_CONNECTIONS, markStyle,connectionStyle)
            
            for i,landMark in enumerate(handMarks.landmark):
                xPosition = int( landMark.x * imageWidth)
                yPosition = int( landMark.y * imageHeight)
                landMark0 = handMarks.landmark[0]
                landMark12 = handMarks.landmark[12]
                if landMark12.x-landMark0.x > 0:
                    isRightHand=True
                else:
                    isRightHand=False
                cv2.putText(image,str(i+1),(xPosition-25,yPosition+5),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),2) ##幫點編號
                if isRecording == True:
                    cv2.circle(image,(imageWidth-30,30),radius=10,color=(0,0,255),thickness=-1) #劃出錄影標示
                    RecordData(i+1,xPosition,yPosition,isRightHand)    


while True:
    retval, image = capturedFrame.read()    ##retval代表讀取幀的成功與否
    if not retval:
        break
    
    rgbImage=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)   ##把BGR圖片轉RGB圖片
    result=hands.process(rgbImage)
    imageHeight = image.shape[0]
    imageWidth = image.shape[1]
    
    time.sleep(1 / fps)  # 等待一段时间，以达到目标帧率
    ShowHandsNode()                        
    previousTime=getFpsAndUpdateTime(previousTime)
    DrawHandMarks()

    cv2.imshow("Hands Detection", image) ##攝影機畫面

    cv2.setMouseCallback("Hands Detection",onMouse) #滑鼠事件

    if cv2.waitKey(1) == ord('r'):      ##按下r紀錄
        StartRecord()

    if cv2.waitKey(1) == ord('q'):      ##按下q關閉
        break

recordedDataList=SortData(recordedDataList)
print(recordedDataList)
with open('data.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(recordedDataList)
capturedFrame.release()         ##釋放攝像頭物件的資源，以確保在結束程式時停止攝像頭的使用


