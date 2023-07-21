import segmentation_yolo
import os

yolo_weights_path=os.path.join(os.getcwd(),'model','poids','bestPose_Yv7.pt')#,','Utilisateur''auguste.verdier','Desktop','Rat_Tracker','model','yolov5'
yolo_repo=os.path.join(os.getcwd(),'model','yoloEncore','YOLOv7')

yolov7_Pose = segmentation_yolo.load_model(yolo_repo,yolo_weights_path)

video1=os.path.join('data','P2R','pre_lesion','GOPRO 1','GH010119.MP4')
video_path2=os.path.join('data','P2R','pre_lesion','GOPRO 2','GH010263.avi')
video3=os.path.join('data','P2R','pre_lesion','GOPRO 3','GH010035.MP4')
video4=os.path.join('data','P2R','pre_lesion','GOPRO 4','GH010044.MP4')
segmentation_yolo.test(video_path2,2200,yolov7_Pose,2)