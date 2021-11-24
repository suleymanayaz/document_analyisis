### Digital Document Analysis 
##### Requirements
```sybase
pip3 install torchvision
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```
##### Setup 
```sybase
python3 ./demo.py --config-file ../configs/DLA_mask_rcnn_R_50_FPN_3x.yaml  --input "0001.jpg" --output "processImage1.jpg"  --confidence-threshold 0.5 --opts MODEL.WEIGHTS ./model_final_trimmed.pth MODEL.DEVICE cpu
```



