import torch
import torch.nn as nn
import torch.nn.functional as F
# class HighDimProjector(nn.Module):
#     def __init__(self, in_channels: int, hidden_channels: int = 64, cond_channels: int = 1):
#         super().__init__()
#         self.base = nn.Sequential(
#         nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
#         nn.GELU(),
#         nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
#         nn.GELU(),
#         )
#         self.cond = nn.Sequential(
#         nn.Conv2d(cond_channels, hidden_channels, kernel_size=1, bias=False),
#         nn.GELU(),
#         nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=1, bias=False),
#         )
#         self.out = nn.Sequential(
#         nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=False),
#         )
#     def forward(self, x: torch.Tensor, cond: torch.Tensor = None) -> torch.Tensor:
#         h = self.base(x)
#         if cond is None:
#             cond = x.mean(dim=1, keepdim=True)
#         if cond.shape[-2:] != x.shape[-2:]:
#             cond = torch.nn.functional.interpolate(cond, size=x.shape[-2:], mode="bilinear", align_corners=False)
#         gamma_beta = self.cond(cond)
#         gamma, beta = gamma_beta.chunk(2, dim=1)
#         h = h * (1.0 + torch.tanh(gamma)) + beta
#         return self.out(h)

# class FactorDecoder(nn.Module):
#     def __init__(self, in_channels: int, out_channels: int):
#         super().__init__()
#         self.decode = nn.Sequential(
#         nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
#         nn.GELU(),
#         nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
#         )
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.decode(x)

class Invertible1x1Conv(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        # orthogonal init
        q, _ = torch.linalg.qr(torch.randn(channels, channels))
        self.weight = nn.Parameter(q)

    def forward(self, x: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        # x: [B, C, H, W]
        if reverse:
            w = torch.inverse(self.weight)
        else:
            w = self.weight
        w = w.view(w.shape[0], w.shape[1], 1, 1)
        return F.conv2d(x, w)


class AffineCoupling(nn.Module):
    def __init__(self, channels: int, hidden: int = 128):
        super().__init__()
        c1 = channels // 2
        c2 = channels - c1
        self.c1 = c1
        self.c2 = c2
        self.net = nn.Sequential(
            nn.Conv2d(c1, hidden, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden, 2 * c2, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        x1, x2 = x[:, :self.c1], x[:, self.c1:]
        st = self.net(x1)
        s, t = st.chunk(2, dim=1)
        s = torch.tanh(s)  # stabilize scale
        if reverse:
            y2 = (x2 - t) * torch.exp(-s)
        else:
            y2 = x2 * torch.exp(s) + t
        return torch.cat([x1, y2], dim=1)


class _FlowBlock(nn.Module):
    def __init__(self, channels: int, hidden: int):
        super().__init__()
        self.inv1x1 = Invertible1x1Conv(channels)
        self.coupling = AffineCoupling(channels, hidden=hidden)

    def forward(self, x: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        if reverse:
            x = self.coupling(x, reverse=True)
            x = self.inv1x1(x, reverse=True)
        else:
            x = self.inv1x1(x, reverse=False)
            x = self.coupling(x, reverse=False)
        return x


class HighDimProjector(nn.Module):
    """
    P: low-dim -> high-dim via lift + invertible flow
    """
    def __init__(self, in_channels: int, hidden_channels: int = 64, cond_channels: int = 1, n_flow: int = 3):
        super().__init__()
        self.lift = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.flow = nn.ModuleList([_FlowBlock(hidden_channels, hidden=max(64, hidden_channels))
                                   for _ in range(n_flow)])

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None) -> torch.Tensor:
        z = self.lift(x)
        for blk in self.flow:
            z = blk(z, reverse=False)
        return z

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        x = z
        for blk in reversed(self.flow):
            x = blk(x, reverse=True)
        return x


class FactorDecoder(nn.Module):
    """
    P^{-1} + projection to native channels.
    """
    def __init__(self, in_channels: int, out_channels: int, n_flow: int = 3):
        super().__init__()
        self.flow = nn.ModuleList([_FlowBlock(in_channels, hidden=max(64, in_channels))
                                   for _ in range(n_flow)])
        self.project_out = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = z
        for blk in reversed(self.flow):
            x = blk(x, reverse=True)
        return self.project_out(x)