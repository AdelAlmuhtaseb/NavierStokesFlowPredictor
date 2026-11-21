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

class SpectralConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, modes1, modes2):
        super().__init__()
        self.out_ch = out_ch
        self.m1, self.m2 = modes1, modes2
        scale = 1 / (in_ch * out_ch)
        self.w1 = nn.Parameter(scale * torch.rand(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.rand(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat))

    def forward(self, x):
        B, _, H, W = x.shape
        xf = torch.fft.rfft2(x)
        out = torch.zeros(B, self.out_ch, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
        out[:, :, :self.m1, :self.m2] = torch.einsum(
            "bixy,ioxy->boxy", xf[:, :, :self.m1, :self.m2], self.w1)
        out[:, :, -self.m1:, :self.m2] = torch.einsum(
            "bixy,ioxy->boxy", xf[:, :, -self.m1:, :self.m2], self.w2)
        return torch.fft.irfft2(out, s=(H, W))


class FNO(nn.Module):
    def __init__(self, in_ch=1, out_ch=3, width=32, modes=12, layers=4):
        super().__init__()
        self.lift = nn.Conv2d(in_ch + 2, width, 1)   # +2 for x/y coordinates
        self.spec = nn.ModuleList(
            [SpectralConv2d(width, width, modes, modes) for _ in range(layers)])
        self.skip = nn.ModuleList(
            [nn.Conv2d(width, width, 1) for _ in range(layers)])
        self.proj = nn.Sequential(
            nn.Conv2d(width, 64, 1), nn.GELU(), nn.Conv2d(64, out_ch, 1))

    def forward(self, x):
        B, _, H, W = x.shape
        gy = torch.linspace(0, 1, H, device=x.device).view(1, 1, H, 1).expand(B, 1, H, W)
        gx = torch.linspace(0, 1, W, device=x.device).view(1, 1, 1, W).expand(B, 1, H, W)
        x = self.lift(torch.cat([x, gx, gy], 1))
        for s, k in zip(self.spec, self.skip):
            x = nn.functional.gelu(s(x) + k(x))
        return self.proj(x)