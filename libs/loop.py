import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

from torchmetrics import functional as tmf

from .data_utils import CUDAPrefetcher
from .model import Generator, GeneratorLoss
from .model import Discriminator


def train(
    trainloader:CUDAPrefetcher,
    generator:Generator,
    discriminator:Discriminator,
    g_criterion:GeneratorLoss,
    d_criterion:torch.nn.BCELoss,
    g_optimizer:optim.Optimizer,
    d_optimizer:optim.Optimizer,
    device:torch.device,
    epoch:int,
    total_epochs:int) -> None:
    
    print(f"# start train {epoch}/{total_epochs}")
    generator.train()
    discriminator.train()
    epoch_start = time.time()
    iter = 0
    trainloader.reset()
    batch_data = trainloader.next()
    while batch_data is not None:
        iter_start = time.time()
        lr = batch_data["lr"].to(device=device, memory_format=torch.channels_last, non_blocking=True)
        hr = batch_data["hr"].to(device=device, memory_format=torch.channels_last, non_blocking=True)
        batch_size = lr.size(0)
        
        with torch.no_grad():
            sr = generator(lr)
        d_out_real = discriminator(hr)
        d_out_fake = discriminator(sr.detach())
        
        # discriminator loss
        d_loss_real = d_criterion(
            d_out_real, torch.ones_like(d_out_real)
        )
        d_loss_fake = d_criterion(
            d_out_fake, torch.zeros_like(d_out_fake)
        )
        d_loss_total = d_loss_real + d_loss_fake
        discriminator.zero_grad()
        d_loss_total.backward()
        d_optimizer.step()
        
        # update generator
        generator.zero_grad()
        sr = generator(lr)
        d_out_fake = discriminator(sr)
        
        g_loss = g_criterion(
            d_out_fake, sr, hr
        )
        g_loss.backward()
        g_optimizer.step()
        
        iter_end = time.time() - iter_start
        # display something
        batch_data = trainloader.next()
        
        
def val(
    valloader:CUDAPrefetcher,
    generator:Generator,
    discriminator:Discriminator,
    device:torch.device,
    epoch:int,
    total_epochs:int) -> None:
    
    print(f"# start val {epoch}/{total_epochs}")
    generator.eval()
    discriminator.eval()
    epoch_start = time.time()
    with torch.no_grad():
        valloader.reset()
        batch_data = valloader.next()
        while batch_data is not None:
            iter_start = time.time()
            lr = batch_data["lr"].to(device=device, memory_format=torch.channels_last, non_blocking=True)
            bi = batch_data["sr"].to(device=device, memory_format=torch.channels_last, non_blocking=True)
            hr = batch_data["hr"].to(device=device, memory_format=torch.channels_last, non_blocking=True)
            batch_size = lr.size(0)
            
            sr = generator(lr)
            
            psnr_bicubic_hr = tmf.peak_signal_noise_ratio(bi.detach(), hr.detach())
            psnr_sr_hr = tmf.peak_signal_noise_ratio(sr.detach(), hr.detach())
            
            ssim_bicubic_hr = tmf.structural_similarity_index_measure(bi, hr)
            ssim_sr_hr = tmf.structural_similarity_index_measure(sr, hr)
            
            iter_end = time.time() - iter_start