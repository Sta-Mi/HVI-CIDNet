import torch
import torch.nn as nn
from net.HVI_transform import RGB_HVI
from net.transformer_utils import *
from net.LCA import *
from net.hdp import HighDimProjector, FactorDecoder
from huggingface_hub import PyTorchModelHubMixin

class CIDNet(nn.Module, PyTorchModelHubMixin):
    def __init__(self, 
                 channels=[36, 36, 72, 144],
                 heads=[1, 2, 4, 8],
                 norm=False,
                 hdp_dim=64
        ):
        super(CIDNet, self).__init__()
        
        
        [ch1, ch2, ch3, ch4] = channels
        [head1, head2, head3, head4] = heads
        
        # HV_ways
        self.HVE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(3, ch1, 3, stride=1, padding=0,bias=False)
            )
        self.HVE_block1 = NormDownsample(ch1, ch2, use_norm = norm)
        self.HVE_block2 = NormDownsample(ch2, ch3, use_norm = norm)
        self.HVE_block3 = NormDownsample(ch3, ch4, use_norm = norm)
        
        self.HVD_block3 = NormUpsample(ch4, ch3, use_norm = norm)
        self.HVD_block2 = NormUpsample(ch3, ch2, use_norm = norm)
        self.HVD_block1 = NormUpsample(ch2, ch1, use_norm = norm)
        self.HVD_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 2, 3, stride=1, padding=0,bias=False)
        )
        
        
        # I_ways
        self.IE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(1, ch1, 3, stride=1, padding=0,bias=False),
            )
        self.IE_block1 = NormDownsample(ch1, ch2, use_norm = norm)
        self.IE_block2 = NormDownsample(ch2, ch3, use_norm = norm)
        self.IE_block3 = NormDownsample(ch3, ch4, use_norm = norm)
        
        self.ID_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.ID_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.ID_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.ID_block0 =  nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 1, 3, stride=1, padding=0,bias=False),
            )
        
        self.HV_LCA1 = HV_LCA(ch2, head2)
        self.HV_LCA2 = HV_LCA(ch3, head3)
        self.HV_LCA3 = HV_LCA(ch4, head4)
        self.HV_LCA4 = HV_LCA(ch4, head4)
        self.HV_LCA5 = HV_LCA(ch3, head3)
        self.HV_LCA6 = HV_LCA(ch2, head2)

        self.hdp_i = HighDimProjector(in_channels=1, hidden_channels=hdp_dim)
        self.hdp_c = HighDimProjector(in_channels=2, hidden_channels=hdp_dim)
        self.hdp_i_enh = HighDimProjector(in_channels=hdp_dim, hidden_channels=hdp_dim)
        self.hdp_c_enh = HighDimProjector(in_channels=hdp_dim, hidden_channels=hdp_dim)
        self.hdp_cross = nn.Conv2d(hdp_dim, hdp_dim, kernel_size=1, bias=False)
        self.hdp_i_decode = FactorDecoder(in_channels=hdp_dim, out_channels=1)
        self.hdp_c_decode = FactorDecoder(in_channels=hdp_dim, out_channels=2)
        self.low_i_enh = nn.Sequential(
            nn.Conv2d(1, ch1, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(ch1, 1, kernel_size=1, bias=False),
        )
        self.low_c_enh = nn.Sequential(
            nn.Conv2d(2, ch1, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(ch1, 2, kernel_size=1, bias=False),
        )
        
        self.I_LCA1 = I_LCA(ch2, head2)
        self.I_LCA2 = I_LCA(ch3, head3)
        self.I_LCA3 = I_LCA(ch4, head4)
        self.I_LCA4 = I_LCA(ch4, head4)
        self.I_LCA5 = I_LCA(ch3, head3)
        self.I_LCA6 = I_LCA(ch2, head2)
        
        self.trans = RGB_HVI()
        
    def forward(self, x, return_aux=False):
        dtypes = x.dtype
        hvi_base = self.trans.HVIT(x)
        i_base = hvi_base[:,2,:,:].unsqueeze(1).to(dtypes)
        c_base = hvi_base[:,:2,:,:].to(dtypes)

        z_i = self.hdp_i(i_base)
        z_c = self.hdp_c(c_base)

        z_i_star, z_c_star = self._orthogonalize(z_i, z_c)
        z_i_hat = self.hdp_i_enh(z_i_star) + z_i_star
        z_c_gate = torch.sigmoid(self.hdp_cross(z_i_star))
        z_c_hat = self.hdp_c_enh(z_c_star) * z_c_gate + z_c_star

        i_from_hdp = i_base + self.hdp_i_decode(z_i_hat)
        c_from_hdp = c_base + self.hdp_c_decode(z_c_hat)
        i_direct = i_base + self.low_i_enh(i_base)
        c_direct = c_base + self.low_c_enh(c_base)

        i = i_from_hdp
        c = c_from_hdp
        hvi = torch.cat([c, i], dim=1)
        # low
        i_enc0 = self.IE_block0(i)
        i_enc1 = self.IE_block1(i_enc0)
        hv_0 = self.HVE_block0(hvi)
        hv_1 = self.HVE_block1(hv_0)
        i_jump0 = i_enc0
        hv_jump0 = hv_0
        
        i_enc2 = self.I_LCA1(i_enc1, hv_1)
        hv_2 = self.HV_LCA1(hv_1, i_enc1)
        v_jump1 = i_enc2
        hv_jump1 = hv_2
        i_enc2 = self.IE_block2(i_enc2)
        hv_2 = self.HVE_block2(hv_2)
        
        i_enc3 = self.I_LCA2(i_enc2, hv_2)
        hv_3 = self.HV_LCA2(hv_2, i_enc2)
        v_jump2 = i_enc3
        hv_jump2 = hv_3
        i_enc3 = self.IE_block3(i_enc2)
        hv_3 = self.HVE_block3(hv_2)
        
        i_enc4 = self.I_LCA3(i_enc3, hv_3)
        hv_4 = self.HV_LCA3(hv_3, i_enc3)
        
        i_dec4 = self.I_LCA4(i_enc4,hv_4)
        hv_4 = self.HV_LCA4(hv_4, i_enc4)
        
        hv_3 = self.HVD_block3(hv_4, hv_jump2)
        i_dec3 = self.ID_block3(i_dec4, v_jump2)
        i_dec2 = self.I_LCA5(i_dec3, hv_3)
        hv_2 = self.HV_LCA5(hv_3, i_dec3)
        
        hv_2 = self.HVD_block2(hv_2, hv_jump1)
        i_dec2 = self.ID_block2(i_dec3, v_jump1)
        
        i_dec1 = self.I_LCA6(i_dec2, hv_2)
        hv_1 = self.HV_LCA6(hv_2, i_dec2)
        
        i_dec1 = self.ID_block1(i_dec1, i_jump0)
        i_dec0 = self.ID_block0(i_dec1)
        hv_1 = self.HVD_block1(hv_1, hv_jump0)
        hv_0 = self.HVD_block0(hv_1)
        
        output_hvi = torch.cat([hv_0, i_dec0], dim=1) + hvi
        output_rgb = self.trans.PHVIT(output_hvi)

        if return_aux:
            cycle_hvi = torch.cat([self.hdp_c_decode(z_c_star), self.hdp_i_decode(z_i_star)], dim=1)
            cycle_rgb = self.trans.PHVIT(cycle_hvi)
            aux = {
                "z_i": z_i,
                "z_c": z_c,
                "z_i_star": z_i_star,
                "z_c_star": z_c_star,
                "z_i_hat": z_i_hat,
                "z_c_hat": z_c_hat,
                "i_base": i_base,
                "c_base": c_base,
                "i_enh": output_hvi[:, 2:3, :, :],
                "c_enh": output_hvi[:, :2, :, :],
                "i_from_hdp": i_from_hdp,
                "c_from_hdp": c_from_hdp,
                "i_direct": i_direct,
                "c_direct": c_direct,
                "cycle_rgb": cycle_rgb,
            }
            return output_rgb, aux

        return output_rgb
    
    def HVIT(self,x):
        hvi = self.trans.HVIT(x)
        return hvi

    @staticmethod
    def _orthogonalize(z_i, z_c, eps=1e-8):
        z_i_norm = z_i / (torch.sqrt((z_i ** 2).sum(dim=1, keepdim=True) + eps))
        projection = (z_c * z_i_norm).sum(dim=1, keepdim=True)
        z_c_ortho = z_c - projection * z_i_norm
        return z_i_norm, z_c_ortho
    
    
