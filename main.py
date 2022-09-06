import os
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import libs
from libs.core_utils import *
from libs.data_utils import *
from libs.my_string import *
from libs.sr_baseline import *
from libs.sr_swinir import *
from libs.tool import *

parser = argparse.ArgumentParser(
    description= "srgan v3"
)
parser.add_argument(
    "--mode", type=str, default="baseline"
)
parser.add_argument(
    "--batch_size", type=int, default=64
)
parser.add_argument(
    "--train_dirs", type=str, default="../data/train"
)
parser.add_argument(
    "--val_dirs", type=str, default="../data/val"
)
parser.add_argument(
    "--upscale_factor", type=int, default=4
)
parser.add_argument(
    "--crop_size", type=int, default=88
)
parser.add_argument(
    "--num_epochs", type=int, default=20
)
parser.add_argument(
    "--save_interval", type=int, default=10
)
parser.add_argument(
    "--save_path", type=str, default="./weights"
)

torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    opt = parser.parse_args()
    
    if not os.path.exists(opt.save_path):
        os.mkdir(opt.save_path)
    save_path = opt.save_path
    
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
    if opt.mode == "baseline":
        generator = Generator(
            scale_factor = opt.upscale_factor
        )
    elif opt.mode == "swinir":
        generator = getSwinIR(
            upscale_factor = opt.upscale_factor
        )
    else:
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
    
    main_start = time.time()
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
        if epoch % opt.save_interval == 0:
            g_name = f"g_{epoch}_{opt.num_epochs}.pth"
            d_name = f"d_{epoch}_{opt.num_epochs}.pth"
            g_name = os.path.join(save_path, g_name)
            d_name = os.path.join(save_path, d_name)
            savemodel(
                generator,
                discriminator,
                g_optimizer,
                d_optimizer,
                g_name,
                d_name
            )
    
    print(f"경과시간 : {time.time() - main_start:.1f}sec")
    exit()