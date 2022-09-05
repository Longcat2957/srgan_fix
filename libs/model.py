import torch
import math
import torch.nn as nn
from torchvision import models as models
from torchvision.models import VGG16_Weights


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm2d(
            channels
        )
        self.prelu = nn.PReLU()     # 파라메트릭 렐루, LeakyRELU와는 달리 음수 기울기 학습 가능
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(
            channels
        )

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        return x + residual
    
class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1
        )
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()
    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


# 생성기 네트워크의 정의
class Generator(nn.Module):
    def __init__(self, scale_factor):
        upsample_block_num = int(
            math.log(scale_factor, 2)
        )
        super(Generator, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(
                3, 64, kernel_size=9, padding=4
            ),
            nn.PReLU()
        )
        
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        
        self.block7 = nn.Sequential(
            nn.Conv2d(
                64, 64, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(64)
        )
        block8 = [UpsampleBlock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(
            64, 3, kernel_size=9, padding=4
        ))
        self.block8 = nn.Sequential(*block8)
    
    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)       # Don't forget Skip-Connection!

        return torch.sigmoid(block8)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))

# 정규화 항의 계산
class L2Loss(nn.Module):
    def __init__(self, l2_loss_weight=1):
        super(L2Loss, self).__init__()
        self.l2_loss_weight = l2_loss_weight

    def forward(self, x):
        batch_size, _, h_x, w_x = x.size()
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_l2 = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x -1, :]), 2).sum()
        w_l2 = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x -1]), 2).sum()
        return self.l2_loss_weight * 2 * (
            h_l2 / count_h + 
            w_l2 / count_w
        ) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]

class GeneratorLoss(nn.Module):
    """
    l_pixel + 1e-3 * l_adv + 6e-3 * l_vgg + 2e-8 * l_reg  
    ------------------------------------------  
    Z : 저해상도(LR) 이미지  
    x = G(z) : 생성기가 제공한 초 해상도(SR)이미지  
    y : 실제 고해상도(HR) 이미지  

    ------------------------------------------ 
    l_adv : 이전 GAN 모델과 유사한 적대적 손실  
    l_pixel : SR과 HR 이미지 간의 MSE 손실인 픽셀 단위 콘텐츠 손실  
    l_vgg : SR및 HR 이미지에서 사전 학습된 VGG 네트워크의 마지막 자질 맵 간 MSE 손실
    l_reg : 정규화 손실(가로 및 세로 방향의 픽셀 기울기의 평균 L2-노름의 합)  
    """
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        vgg  = models.vgg16(weights=VGG16_Weights.DEFAULT)        # vgg16은 31개의 레이어로 구성되어 있음
        loss_network = nn.Sequential(
            *list(vgg.features)[:31]
        ).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.l2_loss = L2Loss()

    def forward(self, out_labels, out_images, target_images):
        # adversarial Loss
        adversarial_loss = torch.mean(1 - out_labels)
        # vgg Loss
        vgg_loss = self.mse_loss(
            self.loss_network(out_images), self.loss_network(target_images)
        )
        # pixel-wise Loss
        pixel_loss = self.mse_loss(
            out_images, target_images
        )
        # regularization Loss
        reg_loss = self.l2_loss(out_images)
        return pixel_loss + 1e-3 * adversarial_loss + 6e-3 * vgg_loss + 2e-8 * reg_loss
