import os
from PIL import Image

import torch
from torch.utils.data import Dataset

import torchvision
from torchvision.transforms import Compose, Resize, ToTensor, \
    RandomCrop
import torchvision.transforms.functional as tvtf

class TrainDataset(Dataset):
    def __init__(self,
                 img_dir:str,
                 crop_size:int,
                 upscale_factor:int
                 ) -> None:
        super(TrainDataset, self).__init__()
        
        assert os.path.isdir(img_dir) and len(os.listdir(img_dir)) > 0
        assert (crop_size // upscale_factor) * upscale_factor == crop_size
        self.filenames = [os.path.join(img_dir, x) for x in os.listdir(img_dir)]
        self.random_crop = Compose([
            RandomCrop(size=crop_size)
        ])
        self.hr_tfms = Compose([
            ToTensor()
        ])
        self.lr_tfms = Compose([
            Resize(
                size = crop_size // upscale_factor,
                interpolation=torchvision.transforms.InterpolationMode.BICUBIC
            ),
            ToTensor()
        ])
        
    def __len__(self):
        return len(self.filenames)
        
    def __getitem__(self, idx:int) -> dict():
        img_obj = Image.open(
            self.filenames[idx]
        )
        random_cropped = self.random_crop(img_obj)
        hr_tensor = self.hr_tfms(random_cropped)
        lr_tensor = self.lr_tfms(random_cropped)
        
        return {"lr" : lr_tensor, "hr" : hr_tensor}

class ValDataset(Dataset):
    def __init__(self,
                 img_dir:str,
                 upscale_factor:int) -> None:
        super(ValDataset, self).__init__()
        assert os.path.isdir(img_dir) and len(os.listdir(img_dir)) > 0
        self.filenames = [os.path.join(img_dir, x) for x in os.listdir(img_dir)]
        self.upscale_factor = upscale_factor
    
    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx:int) -> dict:
        img_obj = Image.open(
            self.filenames[idx]
        )
        w, h = img_obj.size
        crop_size = (min(w, h) // self.upscale_factor) * self.upscale_factor
        center_cropped = tvtf.center_crop(
            img=img_obj, output_size=crop_size
        )
        hr_tensor = tvtf.to_tensor(center_cropped)
        lr_obj = tvtf.resize(
            img=center_cropped, size=crop_size // self.upscale_factor,
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC
        )
        sr_obj = tvtf.resize(
            img=lr_obj, size=crop_size,
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC
        )
        sr_tensor, lr_tensor = map(tvtf.to_tensor, (sr_obj, lr_obj))
        
        return {"hr" : hr_tensor, "sr" : sr_tensor, "lr" : lr_tensor}

class CUDAPrefetcher:
    """Use the CUDA side to accelerate data reading.
    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
        device (torch.device): Specify running device.
    """

    def __init__(self, dataloader, device: torch.device):
        self.batch_data = None
        self.original_dataloader = dataloader
        self.device = device

        self.data = iter(dataloader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch_data = next(self.data)
        except StopIteration:
            self.batch_data = None
            return None

        with torch.cuda.stream(self.stream):
            for k, v in self.batch_data.items():
                if torch.is_tensor(v):
                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch_data = self.batch_data
        self.preload()
        return batch_data

    def reset(self):
        self.data = iter(self.original_dataloader)
        self.preload()

    def __len__(self) -> int:
        return len(self.original_dataloader)
