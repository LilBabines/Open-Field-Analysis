import cv2
import moviepy.editor as mpy
import numpy 
import cupy as np
import matplotlib.pyplot as plt

def delay_run(source1,source2,delay,frame_start):
    cap1 = cv2.VideoCapture(source1)
    cap2 = cv2.VideoCapture(source2)
    fps = cap1.get(cv2.CAP_PROP_FPS)
    cap1.set(cv2.CAP_PROP_POS_FRAMES, int(delay*fps)+frame_start)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
    while True:

        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if ret1 and ret2 :

            cv2.imshow('frame1', frame1)
            cv2.imshow('frame2', frame2)

        key = cv2.waitKey(1)  or 0xff
        if key == 27: 
            break

    cap1.release()
    cap2.release()


    cv2.destroyAllWindows()

def run_4(array_source,delays,framestart=500):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cap1 = cv2.VideoCapture(array_source[0])
    cap2 = cv2.VideoCapture(array_source[1])
    cap3 = cv2.VideoCapture(array_source[2])
    cap4 = cv2.VideoCapture(array_source[3])
    fps=cap1.get(cv2.CAP_PROP_FPS)
    
    cap1.set(cv2.CAP_PROP_POS_FRAMES, int(fps*float(delays[0]))+framestart )
    cap2.set(cv2.CAP_PROP_POS_FRAMES, int(fps*float(delays[1]))+framestart )
    cap3.set(cv2.CAP_PROP_POS_FRAMES, int(fps*float(delays[2]))+framestart )
    cap4.set(cv2.CAP_PROP_POS_FRAMES, int(fps*float(delays[3]))+framestart )
    print([ int(fps*float(delays[0]))+framestart , int(fps*float(delays[1]))+framestart , int(fps*float(delays[2]))+framestart , int(fps*float(delays[3]))+framestart ])

    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    ret3, frame3 = cap3.read()
    ret4, frame4 = cap4.read()
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        ret3, frame3 = cap3.read()
        ret4, frame4 = cap4.read()
        #print(cap1.egt(cv2.CAP_PROP_POS_FRAMES))
        

        if ret1 and ret2 and ret3 and ret4:

            cv2.putText(frame1,str((cap1.get(cv2.CAP_PROP_POS_FRAMES) )),(50, 50),font, 1,(0, 255, 255),2,cv2.LINE_4)
            cv2.imshow('source 1', cv2.resize(frame1,(1920//2,1080//2),interpolation=cv2.INTER_AREA))
            cv2.imshow('source 2', cv2.resize(frame2,(1920//2,1080//2),interpolation=cv2.INTER_AREA))
            cv2.imshow('source 3', cv2.resize(frame3,(1920//2,1080//2),interpolation=cv2.INTER_AREA))
            cv2.imshow('source 4', cv2.resize(frame4,(1920//2,1080//2),interpolation=cv2.INTER_AREA))

        key = cv2.waitKey(50)
        if key == 27: 
            break
    print(cap1.get(cv2.CAP_PROP_POS_FRAMES),cap2.get(cv2.CAP_PROP_POS_FRAMES),cap3.get(cv2.CAP_PROP_POS_FRAMES),cap4.get(cv2.CAP_PROP_POS_FRAMES))
    cap1.release()
    cap2.release()
    cap3.release()
    cap4.release()
    cv2.destroyAllWindows()

def delay_run_4(array_source,delays,framestart=300):
    #font = cv2.FONT_HERSHEY_SIMPLEX
    cap1 = cv2.VideoCapture(array_source[0])
    cap2 = cv2.VideoCapture(array_source[1])
    cap3 = cv2.VideoCapture(array_source[2])
    cap4 = cv2.VideoCapture(array_source[3])
    fps=cap1.get(cv2.CAP_PROP_FPS)
    
    cap1.set(cv2.CAP_PROP_POS_FRAMES, int(fps*float(delays[0]))+framestart )
    cap2.set(cv2.CAP_PROP_POS_FRAMES, int(fps*float(delays[1]))+framestart )
    cap3.set(cv2.CAP_PROP_POS_FRAMES, int(fps*float(delays[2]))+framestart )
    cap4.set(cv2.CAP_PROP_POS_FRAMES, int(fps*float(delays[3]))+framestart )
    print([ int(fps*float(delays[0]))+framestart , int(fps*float(delays[1]))+framestart , int(fps*float(delays[2]))+framestart , int(fps*float(delays[3]))+framestart ])
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    ret3, frame3 = cap3.read()
    ret4, frame4 = cap4.read()
    while True:
        key = cv2.waitKey(0)  or 0xff
        if key == 32 : #espace
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            ret3, frame3 = cap3.read()
            ret4, frame4 = cap4.read()
        elif key == ord('a'):
            ret1, frame1 = cap1.read()
        elif key == ord('z'):
            ret2, frame2 = cap2.read()
        elif key == ord('e'):
            ret3, frame3 = cap3.read()

        elif key == ord('r'):
            ret4, frame4 = cap4.read()

        if ret1 and ret2 and ret3 and ret4:

            #cv2.putText(frame,str((cap1.get(cv2.CAP_PROP_POS_FRAMES) )),(50, 50),font, 1,(0, 255, 255),2,cv2.LINE_4)
            cv2.imshow('source 1', cv2.resize(frame1,(frame1.shape[1]//2,frame1.shape[0]//2)))
            cv2.imshow('source 2', cv2.resize(frame2,(frame1.shape[1]//2,frame1.shape[0]//2)))
            cv2.imshow('source 3', cv2.resize(frame3,(frame1.shape[1]//2,frame1.shape[0]//2)))
            cv2.imshow('source 4', cv2.resize(frame4,(frame1.shape[1]//2,frame1.shape[0]//2)))

       
        if key == 27: 
            break
    print(cap1.get(cv2.CAP_PROP_POS_FRAMES),cap2.get(cv2.CAP_PROP_POS_FRAMES),cap3.get(cv2.CAP_PROP_POS_FRAMES),cap4.get(cv2.CAP_PROP_POS_FRAMES))
    cap1.release()
    cap2.release()
    cap3.release()
    cap4.release()
    cv2.destroyAllWindows()

def get_fps(source):
    movie = mpy.VideoFileClip(source)
    sampling_frequency = movie.audio.fps
    return sampling_frequency

def load_audio(source):
    ad=(mpy.AudioFileClip(source)).to_soundarray()[0:40000000,0]
    


    
    return  ad
def get_audio(source):
    
    ad=  mpy.VideoFileClip(source).audio.to_soundarray()[0:40000000,0]
    
    return  ad


def delay2(audio1,audio2,fps):
    audio1_gpu = np.asarray(audio1)
    audio2_gpu = np.asarray(audio2)
    #print(audio1_gpu)
    #print(len(audio2),len(audio1))
    #print(". . . . .  Correlate compute . . . .")
    correlation_gpu = np.correlate(audio2_gpu, audio1_gpu, mode="full")
    #plt.plot(np.asnumpy(correlation_gpu))
    
    #correlation=signal.correlate(audio2,audio1,mode="full")
    del audio1_gpu,audio2_gpu
    #correlation = np.asnumpy(correlation_gpu)

    argmax = np.argmax(correlation_gpu)
    #plt.show()

    delay1 = audio1.shape[0] - argmax
    sec_delay=delay1/ fps

    return float(sec_delay)
    
def delay(audio1,audio2,fps):

    audio1_gpu = np.asarray(audio1)
    audio2_gpu = np.asarray(audio2)
    #print(audio1_gpu)
    #print(len(audio2),len(audio1))
    print(". . . . .  Correlate compute . . . .")
    correlation_gpu = np.correlate(audio2_gpu, audio1_gpu, mode="full")
    #plt.plot(np.asnumpy(correlation_gpu))
    
    #correlation=signal.correlate(audio2,audio1,mode="full")
    del audio1_gpu,audio2_gpu
    #correlation = np.asnumpy(correlation_gpu)

    argmax = np.argmax(correlation_gpu)
    #plt.show()

    delay1 = audio1.shape[0] - argmax
    sec_delay=delay1/ fps
    print(sec_delay)
    if delay1>0:
        return float(sec_delay),0
    else :
        return 0,abs(float(sec_delay))


def global_delay2(audios,fps):
    print('.    .   .  . ..... COMPUTE DELAYS  .... .  .   .    .')
    total_delay=[0,0,0,0]
    for i in range(len(audios)-1):
        #print(f" . . . . Delay {i} . . . .")
        #print(total_delay)
        d=delay2(audios[0],audios[i+1],fps)

        
        total_delay[i+1]=-d
    m=abs(min(total_delay))
    total_delay=[t+m for t in total_delay]

    return total_delay
def global_delay(audios,fps):
    
    total_delay=[0,0,0,0]
    for i in range(len(audios)-1):
        print(f" . . . . Delay {i} . . . .")
        print(total_delay)
        delay1,delay2=delay(audios[i],audios[i+1],fps)

        if delay1>0:
            print("oui")
            total_delay[i]+=delay1
            for j in range(i):
                total_delay[j]+= delay1
        else :
            print("non")
            total_delay[i+1]+=delay2
    return total_delay

def syncko(sources):
    print("audio1")


    audio1=load_audio(sources[0])
 
    print("audio2")
    audio2=get_audio(sources[1])
    print("audio3")
    audio3=get_audio(sources[2])
    print("audio4")
    audio4=get_audio(sources[3])

    fps=get_fps(sources[0])
    print("delays")
    delays=global_delay([audio1,audio2,audio3,audio4],fps)
    delays[0]=delays[0]
    delay_run_4(sources,delays,framestart=500)
    return delays

