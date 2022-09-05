import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from libs.model import Generator, Discriminator, GeneratorLoss
from libs.data_utils import TrainDataset, ValDataset, CUDAPrefetcher
from libs.loop import train, val

parser = argparse.ArgumentParser(
    description= "srgan v3"
)
parser.add_argument(
    "--batch_size", type=int, default=64
)
parser.add_argument(
    "--train_dirs", type=str, default="../data/train"
)
parser.add_argument(
    "--val_dirs", type=str, default="../data/train"
)
parser.add_argument(
    "--upscale_factor", type=int, default=4
)
parser.add_argument(
    "--crop_size", type=int, default=88
)
parser.add_argument(
    "--num_epochs", type=int, default=100
)


if __name__ == "__main__":
    opt = parser.parse_args()
    traindataset = TrainDataset(
        img_dir = opt.train_dirs,
        crop_size = opt.crop_size,
        upscale_factor= opt.upscale_factor
    )
    valdataset = ValDataset(
        img_dir = opt.val_dirs,
        upscale_factor = opt.upscale_factor
    )
    
    traindloader = DataLoader(
        dataset = traindataset,
        batch_size = opt.batch_size, shuffle=True,
        pin_memory = True, drop_last= True,
        persistent_workers=True, num_workers = os.cpu_count() - 1
    )
    
    valdloader = DataLoader(
        dataset = valdataset,
        batch_size = 1, shuffle=False,
        pin_memory=True, persistent_workers=True,
        num_workers=os.cpu_count() -1,
        drop_last=False
    )
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    trainloader = CUDAPrefetcher(
        traindloader, device=device
    )
    valloader = CUDAPrefetcher(
        valdloader, device=device
    )
    
    generator = Generator(
        scale_factor = opt.upscale_factor
    )
    discriminator = Discriminator()
    g_criterion = GeneratorLoss()
    d_criterion = nn.MSELoss()
    
    # load to device
    generator, discriminator = generator.to(device), discriminator.to(device)
    g_criterion, d_criterion = g_criterion.to(device), d_criterion.to(device)
    
    g_optimizer = torch.optim.Adam(
        generator.parameters(), 
    )
    d_optimizer = torch.optim.Adam(
        discriminator.parameters(),
    )
    
    
    for epoch in range(1, opt.num_epochs + 1):
        train(
            trainloader,
            generator,
            discriminator,
            g_criterion,
            d_criterion,
            g_optimizer,
            d_optimizer,
            device,
            epoch,
            opt.num_epochs
        )
        val(
            valloader,
            generator,
            discriminator,
            device,
            epoch,
            opt.num_epochs
        )
    