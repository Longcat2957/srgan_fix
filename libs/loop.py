import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim

from torchmetrics import functional as tmf

from .data_utils import CUDAPrefetcher
from .model import Generator, GeneratorLoss
from .model import Discriminator
from .string import myprinter, trainstring, valstring


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

    generator.train()
    discriminator.train()
    epoch_start = time.time()
    # display
    mp = myprinter(
        color="GREEN", back="BLACK", style="BRIGHT", indent=1
    )
    ts = trainstring(epoch, total_epochs, trainloader.original_dataloader.dataset.__len__())
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
        ts.update(
            d_loss_total, g_loss, d_out_real, d_out_fake, batch_size, iter_end
        )
        string = ts.getstring()
        mp(string)
        batch_data = trainloader.next()
        
        
def val(
    valloader:CUDAPrefetcher,
    generator:Generator,
    discriminator:Discriminator,
    device:torch.device,
    epoch:int,
    total_epochs:int) -> None:
    mp = myprinter(
        color="YELLOW", back="BLACK", style="BRIGHT", indent=1
    )
    vs = valstring(epoch, total_epochs, valloader.original_dataloader.dataset.__len__())
    generator.eval()
    discriminator.eval()
    epoch_start = time.time()
    valloader.reset()
    batch_data = valloader.next()
    while batch_data is not None:
        with torch.no_grad():
            iter_start = time.time()
            lr = batch_data["lr"].to(device=device, memory_format=torch.channels_last, non_blocking=True)
            bi = batch_data["sr"].to(device=device, memory_format=torch.channels_last, non_blocking=True)
            hr = batch_data["hr"].to(device=device, memory_format=torch.channels_last, non_blocking=True)
            batch_size = lr.size(0)
            
            sr = generator(lr)
            bicubic_result = vs.calculate(
                bi, hr
            )
            sr_result = vs.calculate(
                sr, hr
            )
            iter_end = time.time() - iter_start
            
            vs.update(
                bicubic_result["psnr"], bicubic_result["ssim"],
                sr_result["psnr"], sr_result["ssim"],
                batch_size, iter_end
            )
            string = vs.getstring()
            mp(string)
            batch_data = valloader.next()


def savemodel(generator:nn.Module, discriminator:nn.Module,
              g_optimizer:torch.optim.Optimizer, d_optimizer:torch.optim.Optimizer,
              g_name:str, d_name:str) -> None:
    g_parameter = generator.state_dict()
    d_parameter = discriminator.state_dict()
    g_optimizer = g_optimizer.state_dict()
    d_optimizer = d_optimizer.state_dict()
    
    g_results = {
        "model_weight" : g_parameter, "optimizer_weight" : g_optimizer
    }
    d_results = {
        "model_weight" : d_parameter, "optimizer_weight" : d_optimizer
    }
    torch.save(g_results, g_name)
    torch.save(d_results, d_name)