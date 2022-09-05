import torch
import time
from typing import Union
import torchvision
# Only for Window
# from colorama import init
from colorama import Fore, Back, Style

# Constants
COLOR = [
    "BLACK", "RED", "GREEN", "YELLOW", "BLUE", "MAGENTA",
    "CYAN", "WHITE", "RESET"
]

# FOREGROUND
FORE_COLOR = [
    "Fore." + x for x in COLOR
]
# BACKGROUNG
BACK_COLOR = [
    "Back." + x for x in COLOR
]
class colorprinter(object):
    def __init__(self,
                 color:Union[int, str]="YELLOW",) -> None:
        self.color = color
        self.buffer = None
    
    def __del__(self):
        print("")   # print empty line
    
    def update(self, message:str):
        if not isinstance(message, str):
            try:
                message = str(message)
            except:
                raise Exception("Message를 String으로 바꿀 수 없습니다")
        self.buffer = message
        self.loop()

    def reset(self) -> None:
        self.buffer = None
        
    def loop(self) -> None:
        if self.buffer is not None:
            colorprint(self.buffer, self.color, True)

def colorprint(message:str,
               color:Union[int, str]=-2,
               clear:Union[bool, float, int]=True) -> None:

    if isinstance(color, int):
        try:
            c = FORE_COLOR[color]
        except:
            c = FORE_COLOR[-2]
    elif isinstance(color, str):
        if color in COLOR:
            c = "Fore." + color
        else:
            c = FORE_COLOR[-2]
    else:
        c = FORE_COLOR[-2]
    
    message = " " + message
    if isinstance(clear, bool):
        if clear == True:
            print(eval(c), message, Style.RESET_ALL, end="\r")
        elif clear == False:
            print(eval(c), message, Style.RESET_ALL, end="\n")
            return
    
    elif isinstance(clear, float) or isinstance(clear, int):
        print(eval(c), message, Style.RESET_ALL, end="\r")
        time.sleep(clear)
    return
    
def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


if __name__ == "__main__":
    colorprint("hello world", "RED", 2)
    cpc = colorprinter()
    cpc.update("hellow")
    time.sleep(3)
    cpc.reset()