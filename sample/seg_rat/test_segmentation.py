import segmentation 
import os



source_model=os.path.join("model","UNet_OF_tracking_V2")
source_video=os.path.join("data","GOPRO 1","GH010110.MP4")
model=segmentation.load_model(source_model) 
segmentation.test_segmentation(source_video,model)