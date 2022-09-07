import torch
import torchmetrics.functional as tmff
import sys
from colorama import Fore, Back, Style

FORE_CLASS = [
    "BLACK", "RED", "GREEN", "YELLOW", "BLUE", "MAGENTA",
    "CYAN", "WHITE", "RESET"
]
BACK_CLASS = [
    "BLACK", "RED", "GREEN", "YELLOW", "BLUE", "MAGENTA",
    "CYAN", "WHITE", "RESET"
]
STYLE_CLASS = [
    "DIM", "NORMAL", "BRIGHT",
    "RESET_ALL" # Style.RESET_ALL resets foreground, background, and brightness. 
]

class myprinter(object):
    def __init__(self,
                 color:str="WHITE",
                 back:str="BLACK",
                 style:str="NORMAL",
                 indent:int=1):
        self.data = None
        self.indent = indent
        self.color = color if color != "WHITE" and color in FORE_CLASS else "WHITE"
        self.back = back if back != "BLACK" and back in BACK_CLASS else "BLACK"
        self.style = style if style != "NORMAL" and style in STYLE_CLASS else "NORMAL"
    
    def _make_string(self, string:str) -> None:
        self.data = string
        string = eval(f"Fore.{self.color}")+\
            eval(f"Back.{self.back}")+\
            eval(f"Style.{self.style}")+" "*self.indent+\
            string+\
            Style.RESET_ALL
        string += "\r"
        sys.stdout.write(string)
    
    def clear(self) -> None:
        sys.stdout.flush()
    
    def __call__(self, string:str) -> None:
        if self.data is None:
            self._make_string(string)
        else:
            # if self.data is not None
            sys.stdout.flush()
            self._make_string(string)
    
    def __del__(self) -> None:
        print("\n", end="")
        sys.stdout.flush()
        
# AverageMeter
# https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class trainstring(object):
    def __init__(self, epoch:int, num_epochs:int, datalength:int):
        self.d_loss = AverageMeter()
        self.g_loss = AverageMeter()
        self.d_score = AverageMeter()
        self.g_score = AverageMeter()
        self.itermeter = AverageMeter()
        self.batchmeter = AverageMeter()
        self.head = f"# [{epoch}/{num_epochs}] TRAIN "
        self.datalength = datalength
    
    def update(self,d_loss, g_loss,
               d_score, g_score, batch_size, iter_time
               ) -> None:
        self.d_loss.update(d_loss.detach().cpu().mean(), batch_size)
        self.g_loss.update(g_loss.detach().cpu().mean(), batch_size)
        self.d_score.update(d_score.detach().cpu().mean(), batch_size)
        self.g_score.update(g_score.detach().cpu().mean(), batch_size)
        self.itermeter.update(batch_size/iter_time)
        self.batchmeter.update(batch_size, 1)

    def getstring(self) -> str:
        string = self.head + f"Loss_D : {self.d_loss.avg:.4f}" + \
            " "+f"Loss_G : {self.g_loss.avg:.4f}" + \
            " "+f"D(x) : {self.d_score.avg:.4f}" + \
            " "+f"D(G(z)) : {self.g_score.avg:.4f}" + \
            "\t"+f"{self.itermeter.avg:.2f} iter/sec" + \
                f"\t[{self.batchmeter.sum:.0f}/{self.datalength}]"
        return string

class valstring(object):
    def __init__(self, epoch:int, num_epochs:int, datalength):
        self.bc_psnr = AverageMeter()
        self.bc_ssim = AverageMeter()
        self.sr_psnr = AverageMeter()
        self.sr_ssim = AverageMeter()
        self.itermeter = AverageMeter()
        self.head = f"# [{epoch}/{num_epochs}] VAL "
        self.datalength = datalength
        self.batchmeter = AverageMeter()
    @staticmethod
    def calculate(preds:torch.Tensor, target:torch.Tensor) -> dict:
        psnr = tmff.peak_signal_noise_ratio(
            preds = preds.detach().cpu(), target = target.detach().cpu()
        )
        ssim = tmff.structural_similarity_index_measure(
            preds = preds.detach().cpu(), target = target.detach().cpu()
        )
        return {"psnr" : psnr, "ssim" : ssim}

    def update(self, bc_psnr:torch.Tensor, bc_ssim:torch.Tensor,
               sr_psnr:torch.Tensor, sr_ssim:torch.Tensor,
               batch_size:int, iter_time) -> None:
        self.bc_psnr.update(
            bc_psnr.detach().cpu().mean(), batch_size
        )
        self.bc_ssim.update(
            bc_ssim.detach().cpu().mean(), batch_size
        )
        self.sr_psnr.update(
            sr_psnr.detach().cpu().mean(), batch_size
        )
        self.sr_ssim.update(
            sr_ssim.detach().cpu().mean(), batch_size
        )
        self.itermeter.update(
            batch_size / iter_time
        )
        self.batchmeter.update(
            batch_size, 1
        )
    
    def getstring(self) -> str:
        string = self.head +"(bicubic) " f"psnr : {self.bc_psnr.avg:.4f}" + \
            " "+f"ssim : {self.bc_ssim.avg:.4f}" + \
            "\t(sr) " + \
            " "+f"psnr : {self.sr_psnr.avg:.4f}" + \
            " "+f"ssim : {self.sr_ssim.avg:.4f}" + \
            "\t"+f"{self.itermeter.avg:.2f} iter/sec"+ \
                f"\t[{self.batchmeter.sum:.0f}/{self.datalength}]"
        return string

if __name__ == "__main__":
    import time
    mp = myprinter(color="RED", back="WHITE", style="BRIGHT",
                   indent=3)
    for i in range(10):
        string = str(i)
        mp.__call__(string)
        time.sleep(0.1)
    
    mp.clear()
    print("done")