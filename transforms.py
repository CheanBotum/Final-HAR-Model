import torchvision.transforms as T

def get_transforms(img_size):
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),   
        T.RandomHorizontalFlip(),                           
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  
        T.RandomRotation(15),                                
        T.RandomAffine(degrees=10, translate=(0.1, 0.1)),   
        T.ToTensor(),                                        
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
    ])
