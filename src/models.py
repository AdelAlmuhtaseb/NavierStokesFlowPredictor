import torch
import torch.nn as nn

def block(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1), nn.ReLU(),
        nn.Conv2d(o, o, 3, padding=1), nn.ReLU())

class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=3, base=16):
        super().__init__()
        b = base
        self.e1 = block(in_ch, b)
        self.e2 = block(b, b*2)
        self.e3 = block(b*2, b*4)
        self.mid = block(b*4, b*8)
        self.pool = nn.MaxPool2d(2)
        self.u3 = nn.ConvTranspose2d(b*8, b*4, 2, 2)
        self.d3 = block(b*8, b*4)
        self.u2 = nn.ConvTranspose2d(b*4, b*2, 2, 2)
        self.d2 = block(b*4, b*2)
        self.u1 = nn.ConvTranspose2d(b*2, b, 2, 2)
        self.d1 = block(b*2, b)
        self.out = nn.Conv2d(b, out_ch, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        m = self.mid(self.pool(e3))
        d3 = self.d3(torch.cat([self.u3(m), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1)