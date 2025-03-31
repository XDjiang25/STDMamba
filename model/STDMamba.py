import torch
import torch.nn as nn
from layers.Mamba_EncDec import Encoder, EncoderLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np
from mamba_ssm import Mamba,Mamba2
import torch.nn.functional as F
import math



def get_conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    return nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=bias)


def get_bn(channels):
    return nn.BatchNorm1d(channels)

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1,bias=False):
    if padding is None:
        padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module('conv', get_conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))
    result.add_module('bn', get_bn(out_channels))
    return result

def fuse_bn(conv, bn):

    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std

class ReparamLargeKernelConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, groups,
                 small_kernel,
                 small_kernel_merged=False, nvars=3):
        super(ReparamLargeKernelConv, self).__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        # We assume the conv does not change the feature map size, so padding = k//2. Otherwise, you may configure padding as you wish, and change the padding of small_conv accordingly.
        padding = np.array(kernel_size) // 2
        if small_kernel_merged:
            self.lkb_reparam = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=1, groups=groups, bias=True)
        else:
            self.lkb_origin = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                        stride=stride, padding=padding, dilation=1, groups=groups,bias=False)
            if small_kernel is not None:
                assert small_kernel <= kernel_size, 'The kernel size for re-param cannot be larger than the large kernel!'
                self.small_conv = conv_bn(in_channels=in_channels, out_channels=out_channels,
                                            kernel_size=small_kernel,
                                            stride=stride, padding=np.array(small_kernel) // 2, groups=groups, dilation=1,bias=False)


    def forward(self, inputs):

        if hasattr(self, 'lkb_reparam'):
            out = self.lkb_reparam(inputs)
        else:
            out = self.lkb_origin(inputs)
            if hasattr(self, 'small_conv'):
                out += self.small_conv(inputs)

        return out

    def PaddingTwoEdge1d(self,x,pad_length_left,pad_length_right,pad_values=0):

        D_out,D_in,ks=x.shape
        if pad_values ==0:
            pad_left = torch.zeros(D_out,D_in,pad_length_left)
            pad_right = torch.zeros(D_out,D_in,pad_length_right)
        else:
            pad_left = torch.ones(D_out, D_in, pad_length_left) * pad_values
            pad_right = torch.ones(D_out, D_in, pad_length_right) * pad_values
        x = torch.cat([pad_left,x],dims=-1)
        x = torch.cat([x,pad_right],dims=-1)
        return x

    def get_equivalent_kernel_bias(self):

        eq_k, eq_b = fuse_bn(self.lkb_origin.conv, self.lkb_origin.bn)

        if hasattr(self, 'small_conv'):
            small_k, small_b = fuse_bn(self.small_conv.conv, self.small_conv.bn)

            eq_b += small_b

            eq_k += self.PaddingTwoEdge1d(small_k, (self.kernel_size - self.small_kernel) // 2,
                                          (self.kernel_size - self.small_kernel) // 2, 0)
        return eq_k, eq_b

    def merge_kernel(self):
        eq_k, eq_b = self.get_equivalent_kernel_bias()
        self.lkb_reparam = nn.Conv1d(in_channels=self.lkb_origin.conv.in_channels,
                                     out_channels=self.lkb_origin.conv.out_channels,
                                     kernel_size=self.lkb_origin.conv.kernel_size, stride=self.lkb_origin.conv.stride,
                                     padding=self.lkb_origin.conv.padding, dilation=self.lkb_origin.conv.dilation,
                                     groups=self.lkb_origin.conv.groups, bias=True)
        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data = eq_b
        self.__delattr__('lkb_origin')
        if hasattr(self, 'small_conv'):
            self.__delattr__('small_conv')

class ICB(nn.Module):
    def __init__(self, in_features,hidden_features,drop=0.):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features,hidden_features,1)
        self.conv2 = nn.Conv1d(in_features, hidden_features, 3, 1, padding=1)
        self.conv3 = nn.Conv1d(hidden_features,in_features,1)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()
    def forward(self,x):
        x = x.permute(0,2,1)
        x1 = self.conv1(x)
        x1_1 = self.act(x1)
        x1_2 = self.drop(x1_1)
        
        x2 = self.conv2(x)
        x2_1 = self.act(x2)
        x2_2 = self.drop(x2_1)
        
        
        out1 = x1 * x2_2
        out2 = x2 * x1_2

        x = self.conv3(out1 + out2)
        x = x.permute(0,2,1)
        return x


class GD(nn.Module):
    def __init__(self,LDkernel_size=25):
        super(GD, self).__init__()
        #Gaussian initialization
        self.conv=nn.Conv1d(1, 1, kernel_size=GDkernel_size, stride=1, padding=int(LDkernel_size//2), padding_mode='replicate', bias=True) 
        
        kernel_size_half = LDkernel_size // 2
        sigma = 1.0  # 1 for variance stable
        weights = torch.zeros(1, 1, GDkernel_size)
        for i in range(LDkernel_size):
            weights[0, 0, i] = math.exp(-((i - kernel_size_half) / (2 * sigma)) ** 2)

        # Set the weights of the convolution layer
        self.conv.weight.data = F.softmax(weights,dim=-1)
        self.conv.bias.data.fill_(0.0)
        
    def forward(self, inp):
        # Permute the input tensor to match the expected shape for 1D convolution (B, N, T)
        inp = inp.permute(0, 2, 1)
        # Split the input tensor into separate channels
        input_channels = torch.split(inp, 1, dim=1)
        
        # Apply convolution to each channel
        conv_outputs = [self.conv(input_channel) for input_channel in input_channels]
        
        # Concatenate the channel outputs
        out = torch.cat(conv_outputs, dim=1)
        out = out.permute(0, 2, 1)
        return out
    
       
class Model(nn.Module):

#just make sure d_model * expand / headdim = multiple of 8
# headdim = 64
# 128 
    def __init__(self, configs):
        super(Model, self).__init__()
        nvars = 7
        drop = 0.1
        
        self.e_layers = configs.e_layers
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                        Mamba2(
                            d_model=configs.d_model,  # Model dimension d_model
                            d_state=configs.d_state,  # SSM state expansion factor 状态扩展因子
                            d_conv=4,  # Local convolution width
                            expand=4,  # Block expansion factor)
                        ),
                        Mamba2(
                            d_model=configs.d_model,  # Model dimension d_model
                            d_state=configs.d_state,  # SSM state expansion factor
                            d_conv=4,  # Local convolution width
                            expand=4,  # Block expansion factor)
                        ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        
        #DWconv提取时间特征    B N E
        self.dw = ReparamLargeKernelConv(in_channels=configs.d_model, out_channels=configs.d_model,
                                         kernel_size=configs.large_size, stride=1, groups=configs.d_model,
                                         small_kernel=configs.small_size, small_kernel_merged=False, nvars=nvars)
        self.norm = nn.BatchNorm1d(configs.d_model)
        
        #convffn1
        self.ffn1pw1 = nn.Conv1d(in_channels=configs.d_model, out_channels=configs.d_ff, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=4)
        self.ffn1act = nn.GELU()
        self.ffn1pw2 = nn.Conv1d(in_channels=configs.d_ff, out_channels=configs.d_model, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=4)
        self.ffn1drop1 = nn.Dropout(drop)
        self.ffn1drop2 = nn.Dropout(drop)
        
        #icb
        self.icb = ICB(in_features=configs.d_model,hidden_features=128)
        
        self.LD = LD(LDkernel_size=configs.LDkernel_size)
        
        

        
        
        
        #线性层将输入特征从大小为configs.d_model的向量 投影到 大小为configs.pred_len的向量
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)
    # a = self.get_parameter_number()
    #
    # def get_parameter_number(self):
    #     """
    #     Number of model parameters (without stable diffusion)
    #     """
    #     total_num = sum(p.numel() for p in self.parameters())
    #     trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
    #     trainable_ratio = trainable_num / total_num
    #
    #     print('total_num:', total_num)
    #     print('trainable_num:', total_num)
    #     print('trainable_ratio:', trainable_ratio)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape # B L N
   
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        
        enc_out = self.enc_embedding(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens
        
        main = self.LD(enc_out) ## Trend  B N E
        residual = enc_out - main ## Merit  B N E
        end = residual.size(-1)
        resintra = torch.cat([residual[:, :, end:], residual[:, :, 0:end]], dim=-1)
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(residual, attn_mask=None)
        #outintra, attns = self.encoder(resintra,attn_mask=None)
        # B N E -> B N S -> B S N
        #
        

        #
        # output = enc_out # B N E
        # enc_out = enc_out.permute(0,2,1) # B N E -> B E N
        # enc_out = self.dw(enc_out)
        # enc_out = self.norm(enc_out)
        # enc_out = self.ffn1drop1(self.ffn1pw1(enc_out))
        # enc_out = self.ffn1act(enc_out)
        # enc_out = self.ffn1drop2(self.ffn1pw2(enc_out))
        # enc_out = enc_out.permute(0,2,1) # B E N -> B N E
        # enc_out = enc_out + output
        #
        output = enc_out # B N E
        dec_out0 = output + main ##Season + Trend
        dec_out0 = self.projector(dec_out0).permute(0, 2, 1)[:, :, :N]  
        
        enc_out = enc_out.permute(0,2,1) # B N E -> B E N
        #print("enc_out:",enc_out)
        #print("main:",main)
        main = main.permute(0,2,1) # B N E -> B E N
        main = self.dw(main)
        main = self.norm(main)
        #main = main.permute(0,2,1) ## B E N -> B N E
        main = self.ffn1drop1(self.ffn1pw1(main))
        main = self.ffn1act(main)
        main = self.ffn1drop2(self.ffn1pw2(main))
        main = main.permute(0,2,1)
        #enc_out = self.dw(enc_out)
        #enc_out = self.norm(enc_out)
        #enc_out = enc_out.permute(0,2,1) # B E N -> B N E
        #enc_out = self.icb(enc_out)
        #enc_out = enc_out.permute(0,2,1)
        #enc_out = self.ffn1drop1(self.ffn1pw1(enc_out))
        #enc_out = self.ffn1act(enc_out)
        #enc_out = self.ffn1drop2(self.ffn1pw2(enc_out))
        enc_out = enc_out.permute(0,2,1) # B E N -> B N E
        #enc_out = enc_out + output ## no linear Season
                   
        #enc_out = enc_out + outintra                     
        ##projection
        main0 = main ##no linear Trend
        #print("enc_out:",enc_out.size())
        #print("main0:",main0.size())
        dec_out = main + enc_out 
        dec_out = self.projector(dec_out).permute(0, 2, 1)[:, :, :N]       
        #main = self.projector(main).permute(0, 2, 1)[:, :, :N] ##Trend
        #dec_out0 = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates   Season
        #main = main.permute(0, 2, 1)
        #dec_out0 = enc_out.permute(0, 2, 1)
        #print("enc_out:",dec_out0.size())
        #print("main0",main.size())
        
        
        #dec_out = dec_out0 + main
        #dec_out = dec_out.permute(0,2,1)
        #print("dec_out:",dec_out.size())
        #dec_out = self.projector(dec_out)
        #print("dec_out线性:",dec_out.size())
        #dec_out = self.projector(dec_out).permute(0, 2, 1)[:, :, :N]
        #print("dec_out线性2:",dec_out.size())

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out,dec_out0 ##main and season


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out,dec_out0 = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        return dec_out[:, -self.pred_len:, :], dec_out0  # [B, L, D]
