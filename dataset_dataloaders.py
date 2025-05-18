import torch
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from torchvision.transforms import v2
from PIL import Image
from torchvision.transforms.functional import pad
from utils import PadStandard, MultiClassImageFolder, BinaryClassImageFolder, collate_tensor_fn_remove_none


# - - - - - Parameters - - - - - #

TRAIN_DATA_DIR = "./DataSet/cross_out_dataset_prelim/train"
VALIDATION_DATA_DIR = "./DataSet/cross_out_dataset_prelim/val"
TEST_DATA_DIR = "./DataSet/cross_out_dataset_prelim/test"

BATCH_SIZE = 384
# - - - - - Data loading - - - - - #


transformAverage = v2.Compose([
                                v2.Grayscale(1),
                                PadStandard(),
                                v2.ToImage(), 
                                v2.ToDtype(torch.float32, scale=True),
                                v2.Normalize(mean = [0.8883], std = [0.2263]),
                                v2.Resize((80,180)),
                                ])

transformAverageTraining = v2.Compose([
                                v2.Grayscale(1),
                                PadStandard(), 
                                v2.GaussianBlur(kernel_size=21, sigma = (0.5, 1)),
                                v2.ToImage(),
                                v2.ToDtype(torch.float32, scale=True),
                                v2.Normalize(mean = [0.8883], std = [0.2263]),
                                v2.RandomInvert(p = 0.5),
                                v2.GaussianNoise(sigma = 0.1),
                                v2.Resize((80,180)),
                                ])

train_binary = BinaryClassImageFolder(root=TRAIN_DATA_DIR, transform=transformAverage)
val_binary = BinaryClassImageFolder(root=VALIDATION_DATA_DIR, transform=transformAverage)
test_binary = BinaryClassImageFolder(root=TEST_DATA_DIR, transform=transformAverage)

# new testing set derived from trainset #
# Included for effiency reasons
train_binary, _ = torch.utils.data.random_split(train_binary, [1, 0])

train_binary.transform = transformAverageTraining

labels = torch.tensor(train_binary.dataset.get_labels())[train_binary.indices]
class_counts = torch.bincount(labels)
class_weights = 1. / class_counts.float()
sample_weights = class_weights[labels]
binary_training_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(labels), replacement=True)
              
train_binary_loader = DataLoader(train_binary, batch_size=BATCH_SIZE, sampler = binary_training_sampler, num_workers = 4, collate_fn=collate_tensor_fn_remove_none)
val_binary_loader = DataLoader(val_binary, batch_size=BATCH_SIZE, num_workers = 4, collate_fn=collate_tensor_fn_remove_none)
test_binary_loader = DataLoader(test_binary, batch_size=BATCH_SIZE, num_workers = 4, collate_fn=collate_tensor_fn_remove_none)

train_multi = MultiClassImageFolder(root=TRAIN_DATA_DIR, transform=transformAverage)
val_multi = MultiClassImageFolder(root=VALIDATION_DATA_DIR, transform=transformAverage)
test_multi = MultiClassImageFolder(root=TEST_DATA_DIR, transform=transformAverage)

# new testing set derived from trainset #
train_multi, _ = torch.utils.data.random_split(train_multi, [1, 0])

train_multi.transform = transformAverageTraining

train_multi_loader = DataLoader(train_multi, batch_size=BATCH_SIZE, shuffle=True, num_workers = 4, collate_fn=collate_tensor_fn_remove_none)
val_multi_loader = DataLoader(val_multi, batch_size=BATCH_SIZE, shuffle=True, num_workers = 4, collate_fn=collate_tensor_fn_remove_none)
test_multi_loader = DataLoader(test_multi, batch_size=BATCH_SIZE, shuffle=True, num_workers = 4, collate_fn=collate_tensor_fn_remove_none)
    
print("training_loader_size_binary:", len(train_binary_loader), 
      "validation_loader_size_binary:", len(val_binary_loader), 
      "test_loader_size_binary:", len(test_binary_loader))

print("training_loader_size_multi:", len(train_multi_loader), 
      "validation_loader_size_multi:", len(val_multi_loader), 
      "test_loader_size_multi:", len(test_multi_loader),
        test_multi.classes)
