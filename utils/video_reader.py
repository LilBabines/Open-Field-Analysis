import cv2
import os
def readK(source,frame_start=0):

    cap = cv2.VideoCapture(source)
    current_frame=frame_start
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

    font = cv2.FONT_HERSHEY_SIMPLEX
    
    if not cap.isOpened():
        print("Error opening video")

    while(cap.isOpened()):
        
        status, frame = cap.read()
        if status:
        
            
            #cap.set(cap.get(cv2.CAP_PROP_POS_FRAMES),)
            
            #cv2.putText(frame,str((cap.get(cv2.CAP_PROP_POS_FRAMES) -frame_start)//5),(50, 50),font, 1,(0, 255, 255),2,cv2.LINE_4)
            cv2.imshow('frame', frame)
            current_frame+=1
            print(cap.get(cv2.CAP_PROP_POS_FRAMES))
        key = cv2.waitKey(20)
        if key == ord('a'):
            print(current_frame)
            break

    cap.release()
    cv2.destroyAllWindows()


       
def capture(sources,delays,frame_start=700):
    cap1=cv2.VideoCapture(sources[0])
    cap2=cv2.VideoCapture(sources[1])
    cap3=cv2.VideoCapture(sources[2])
    cap4=cv2.VideoCapture(sources[3])

    fps=cap1.get(cv2.CAP_PROP_FPS)
        
    cap1.set(cv2.CAP_PROP_POS_FRAMES, fps*delays[0]+frame_start)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, fps*delays[1]+frame_start)
    cap3.set(cv2.CAP_PROP_POS_FRAMES, fps*delays[2]+frame_start)
    cap4.set(cv2.CAP_PROP_POS_FRAMES, fps*delays[3]+frame_start)

    return cap1,cap2,cap3,cap4
def read(caps):
    ret1,frame1=caps[0].read()
    ret2,frame2=caps[1].read()
    ret3,frame3=caps[2].read()
    ret4,frame4=caps[3].read()
    return frame1,frame2,frame3,frame4, ret1 and ret2 and ret3 and ret4

#source2=os.path.join( "data","P2D","3_semaine_postL","GOPRO 1","GH010181.MP4")

#readK(source2,19600)