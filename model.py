import torch
from torch import nn
from torchsummary import summary

def DoubleConV(in_channels, out_channels):
    mid_channels = out_channels
    return nn.Sequential(
        nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1), # padding = 1, giu nguyen size anh
        nn.BatchNorm2d(mid_channels),
        nn.LeakyReLU(0.2),
        nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2)
    )
# 0 0 0 0 0 0 0
# 0 1 1 1 1 1 0
# 0 1 1 1 1 1 0 
# 0 1 1 1 1 1 0
# 0 0 0 0 0 0 0

def Down(in_channels, out_channels):
    return nn.Sequential(
        nn.MaxPool2d(2),
        DoubleConV(in_channels, out_channels)
    )

def Up(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    )

def OutConv(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.inc = DoubleConV(n_channels, 64) #floor 1
        self.down1 = Down(64,128)
        self.down2 = Down(128,256)
        self.down3 = Down(256,512)
        self.center = Down(512,1024)
        self.up1 = Up(1024,512)
        self.tmp1 = DoubleConV(1024,512)
        self.up2 = Up(512,256)
        self.tmp2 = DoubleConV(512,256)
        self.up3 = Up(256,128)
        self.tmp3 = DoubleConV(256,128)
        self.up4 = Up(128,64)
        self.tmp4 = DoubleConV(128,64)
        self.output = nn.Sequential(
            OutConv(64, n_classes),
            nn.Sigmoid()
        )
    def forward(self, x):
        f1L = self.inc(x) #64
        f2L = self.down1(f1L) #128
        f3L = self.down2(f2L) #256
        f4L = self.down3(f3L) #512
        cent = self.center(f4L) #1024

        f4R = self.up1(cent) #512
        f4RT = self.tmp1(torch.cat((f4L, f4R), dim=1)) #512->1024
        f3R = self.up2(f4RT) #256
        f3RT = self.tmp2(torch.cat((f3L, f3R), dim=1)) #256->512
        f2R = self.up3(f3RT) #128
        f2RT = self.tmp3(torch.cat((f2L, f2R), dim=1)) #128->256
        f1R = self.up4(f2RT) #64
        f1RT = self.tmp4(torch.cat((f1L, f1R), dim=1)) #512->1024
        logits = self.output(f1RT)
        return logits
        
if __name__ == "__main__":
    model = UNet(3,1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    summary(model, input_size=(3, 240, 320))
    