import math
import numpy as np
import torch
import os 
from PIL import Image
from torchvision.transforms.functional import pad
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder


class PadStandard:
    
    """
        Padding implementation used as part of transforms pipeline
    """
    
    def __call__(self, image):
        w, h = image.size
        if (w/h) < 2.:
            padding = (math.ceil((2.25*h - w)/2), 0, math.floor((2.25*h - w)/2), 0)
        else:
            padding = (0, math.ceil((w - 2.25*h)/2), 0, math.floor((w - 2.25*h)/2))
        return pad(image, padding, 255, 'constant')
    

def collate_tensor_fn_remove_none(batch, *, collate_fn_map = None):
    
    """
        Custom collate function used to facilitate conditional behavior in dataset __getitem__ method
    """
    
    batch = [x for x in batch if x is not None]
    if not batch:
        return None

    elem = batch[0]

    if isinstance(elem, tuple):
        transposed = list(zip(*batch))
        return tuple(collate_tensor_fn_remove_none(samples) for samples in transposed)

    if all(isinstance(x, torch.Tensor) for x in batch):
        return torch.stack(batch, dim=0)

    if all(isinstance(x, (int, float, np.integer, np.floating)) for x in batch):
        return torch.tensor(batch)

    if collate_fn_map:
        for key, fn in collate_fn_map.items():
            if isinstance(elem, key):
                return fn(batch)
                
    return batch

class BinaryClassImageFolder(ImageFolder):
    def __init__(self, root, transform, nonwords = None):
        super().__init__(root=root, transform=transform)
        if nonwords != None:
            nonwords_file = open(nonwords, "r")
            self.nonwords = nonwords_file.readlines()
            self.nonwords = [x.strip() for x in self.nonwords]
            nonwords_file.close()

    def find_classes(self, directory):
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
    
        class_to_idx = {cls_name: 0 if cls_name == "CLEAN" else 1 for cls_name in classes}
        return classes, class_to_idx

    def get_labels(self):
        return torch.tensor([y for x, y in self.samples])
    
    def __getitem__(self, index: int):
        path, target = self.samples[index]
        check_path = path.split("/")[-1]
        sample = self.loader(path)
        w, h = sample.size
        
        # Implement conditional behavior here, 
        # return None as needed for examples, examples given below
        
        #if check_path in self.nonwords:
        #    return None
        #if w*h <= 100:
        #    return None
        #if w <= 50 and h <= 50:
        #    return None

        sample = self.transform(sample)

        return sample, target, index
    
class MultiClassImageFolder(ImageFolder):
    def __init__(self, root, transform, nonwords = None):
        super().__init__(root=root, transform=transform)
        if nonwords != None:
            nonwords_file = open(nonwords, "r")
            self.nonwords = nonwords_file.readlines()
            self.nonwords = [x.strip() for x in self.nonwords]
            nonwords_file.close()

    def find_classes(self, directory):
        # 8 class vs 7 class
        
        #classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir() and entry.name != 'CLEAN')
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
    
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index: int):
        
        path, target = self.samples[index]
        check_path = path.split("/")[-1]
        sample = self.loader(path)
        w, h = sample.size
        
        # Implement conditional behavior here, 
        # return None as needed for examples, examples given below
        
        #if check_path in self.nonwords:
        #    return None
        #if w*h <= 1600:
        #    return None
        #if w <= 30 or h <= 30:
        #    return None
        #if w <= 20 and h <= 20:
        #    return None
                
        sample = self.transform(sample)

        return sample, target, index