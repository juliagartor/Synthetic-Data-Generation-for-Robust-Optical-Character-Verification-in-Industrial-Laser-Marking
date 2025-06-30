from ultralytics import YOLO

model = YOLO("/mnt/DADES/home/jgarcia/CODE/2) DIGIT LOCATION AND RECOGNITION/runs/detect/train11/weights/best.pt")


model.train(
    data="/mnt/DADES/home/jgarcia/CODE/2) DIGIT LOCATION AND RECOGNITION/dataset_sdxl/dataset.yaml",
    epochs=500,
    patience=30,  
    batch=16,    
    
    # Learning rate settings 
    lr0=0.003,     
    lrf=0.001,     
    
    # Regularization 
    dropout=0.3,   
    
    # Warmup 📶
    warmup_epochs=5,
    
    # Augmentations 
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    translate=0.2,
    scale=0.5,
    fliplr=0.6,
    mixup=0.25,
    auto_augment="randaugment",  

    # Advanced tricks 
    multi_scale=True,
    mosaic=True,
    cache=True, 

    plots=True
)

