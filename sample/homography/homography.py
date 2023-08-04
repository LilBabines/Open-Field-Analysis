import numpy as np
import cv2 as cv
import os

def findHomography(source):
    cap = cv.VideoCapture(source)
    cap.set(cv.CAP_PROP_POS_FRAMES, 50)
    ret,frame=cap.read()
    while not ret:
        #print("a")
        ret,frame = cap.read()
    drawing = False # true if mouse is pressed
    #src_x, src_y = -1,-1
    #dst_x, dst_y = -1,-1

    src_list = []
    dst_list = []

    # mouse callback function
    def select_points_src(event,x,y,flags,param):
        global src_x, src_y, drawing
        if event == cv.EVENT_LBUTTONDOWN:
            drawing = True
            src_x, src_y = x,y
            cv.circle(src_copy,(x,y),5,(0,0,255),-1)
        elif event == cv.EVENT_LBUTTONUP:
            drawing = False

    # mouse callback function
    def select_points_dst(event,x,y,flags,param):
        global dst_x, dst_y, drawing
        if event == cv.EVENT_LBUTTONDOWN:
            drawing = True
            dst_x, dst_y = x,y
            cv.circle(dst_copy,(x,y),5,(0,0,255),-1)
        elif event == cv.EVENT_LBUTTONUP:
            drawing = False

    def get_plan_view(src, dst):
        global H
        src_pts = np.array(src_list).reshape(-1,1,2)
        dst_pts = np.array(dst_list).reshape(-1,1,2)
        H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
        print("H:")
        print(H)
        plan_view = cv.warpPerspective(src, H, (dst.shape[1], dst.shape[0]))
        return plan_view

    def merge_views(src, dst):
        plan_view = get_plan_view(src, dst)
        for i in range(0,dst.shape[0]):
            for j in range(0, dst.shape[1]):
                if(plan_view.item(i,j,0) == 0 and \
                plan_view.item(i,j,1) == 0 and \
                plan_view.item(i,j,2) == 0):
                    plan_view.itemset((i,j,0),dst.item(i,j,0))
                    plan_view.itemset((i,j,1),dst.item(i,j,1))
                    plan_view.itemset((i,j,2),dst.item(i,j,2))

        return plan_view
    #src = cv.imread('homography-computation\imgs\frame3.png', -1)
    #src = cv.imread(os.path.join("sample","homography","imgs","frame3.png"), -1)
    src_copy = frame.copy()
    cv.namedWindow('src')
    cv.moveWindow("src", 80,80)
    cv.setMouseCallback('src', select_points_src)


    #dst = cv.imread('homography-computation\imgs\ref.png', -1)
    dst = cv.imread(os.path.join("sample","homography","imgs","ref.png"), -1)

    dst_copy = dst.copy()
    cv.namedWindow('dst')
    cv.moveWindow("dst", 780,80)
    cv.setMouseCallback('dst', select_points_dst)


    while(1):
        cv.imshow('src',src_copy)
        cv.imshow('dst',dst_copy)
        k = cv.waitKey(1) & 0xFF
        if k == ord('s'):
            print('save points')
            cv.circle(src_copy,(src_x,src_y),5,(0,255,0),-1)
            cv.circle(dst_copy,(dst_x,dst_y),5,(0,255,0),-1)
            src_list.append([src_x,src_y])
            dst_list.append([dst_x,dst_y])
            print("src points:")
            print(src_list)
            print("dst points:")
            print(dst_list)
        elif k == ord('h'):
            print('create plan view')
            plan_view = get_plan_view(frame, dst)
            cv.imshow("plan view", plan_view) 
        elif k == ord('m'):
            print('merge views')
            merge = merge_views(frame,dst)      
            cv.imshow("merge", merge)        
        elif k == 27:
            cv.destroyAllWindows()
            return H
            
    

def run(path_to_video,frame_start,H):
    '''Lit une video et applique l'homography correspondante'''
    cap = cv.VideoCapture(path_to_video)
    current_frame=frame_start
    cap.set(cv.CAP_PROP_POS_FRAMES, frame_start)
    
    
    if not cap.isOpened():
        print("Error opening video")

    while(cap.isOpened()):
        
        status, frame_rgb = cap.read()
        
        if status:

            plan_view = cv.warpPerspective(frame_rgb, H, (1000, 1000))
            
            cv.imshow('view', plan_view)
            cv.imshow('frame', frame_rgb)
            current_frame+=1
        key = cv.waitKey(3)
        if key ==27:
            print("Final Frame : ",current_frame)
            break

    cap.release()
    cv.destroyAllWindows()


    
    


def get(rat,exp,source,path,i):
    '''
    - load l'homography si elle st trouver dans le dossier data/rat/exp/homographhy/homo{i}.npy
    - si pas trouvée : calcul une nouvelle correspondance de point'''
    path_homo=os.path.join(path,rat,exp,'homography',f'homo{i}.npy')
    if os.path.exists(path_homo):

        return np.load(path_homo)
    else :
        H=findHomography(source)

        #enregistre l'homographie dans le dossier spécifié(format matrix numpy)

        np.save(os.path.join(path,rat,exp,'homography',f"homo{i}.npy"),H)
      
        return H