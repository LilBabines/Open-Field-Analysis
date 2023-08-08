# Box et posture Prédictions (YOLOv5)

## Fonction :

blablabla

## Utilisation :

terminal command (toujours se placer dans le répertoire OpenField pour terminal, sinon ajuster les chemin des fichiers) : 
- python \sample\seg_rat\segmentation_yolo.py --source [video_path]
- python \sample\seg_rat\segmentation_yolo.py --source [video_path] --weights [weights_path]
  
Pour enregistrer une video de prédiction ou filtrer les prediction avec leurs scores de confiance, il faudrait utiliser le fichier .\model\yolov5\detect.py. Le fichier comporte des explicatoin d'utilisation au dbut. Conseil d'utilisatoin :
- python .\model\yolov5\detect.py --weights [path_weights] --source [vdieo_path]
- python .\model\yolov5\detect.py --weights .\model\poids\LAST_LAST_LAST.pt --source .\data\test_rat\test_exp\GOPRO1\****.MP4 

- python .\model\yolov5\detect.py --weights .\model\poids\best_1920.pt --source .\data\test_rat\test_exp\GOPRO1\****.MP4 --img-size 1920 --conf-thres 0.6 --line-thickness 5