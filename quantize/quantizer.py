import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import tqdm
import numpy as np
import pdb
import math

CLIPMIN = 1e-5

class StraightThrough(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input


def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


def lp_loss(pred, tgt, p=2.0, reduction="none"):
    """
    loss function measured in L_p Norm
    """
    if reduction == "none":
        return (pred - tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred - tgt).abs().pow(p).mean()


class UniformAffineQuantizer(nn.Module):
    def __init__(
        self,
        n_bits: int = 8,
        symmetric: bool = False,
        per_channel_axes=[],
        metric="minmax",
        dynamic=False,
        dynamic_method="per_tensor",
        group_size=None,
        shape=None,
        lwc=False,
        disable_zero_point=False,
        # I-LLM
        rescale=False,
        rescale_limit=False,
        lsq = False,
    ):
        """
        support cluster quantize
        dynamic_method support per_token and per_cluster
        """
        super().__init__()
        self.symmetric = symmetric
        self.disable_zero_point = disable_zero_point
        assert 2 <= n_bits <= 16, "bitwidth not supported"
        self.n_bits = n_bits
        if self.disable_zero_point:
            self.qmin = -(2 ** (n_bits - 1))
            self.qmax = 2 ** (n_bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** (n_bits) - 1
        self.per_channel_axes = per_channel_axes
        self.metric = metric
        self.cluster_counts = None
        self.cluster_dim = None

        self.scale = None
        self.zero_point = None
        self.round_zero_point = None

        self.cached_xmin = None
        self.cached_xmax = None
        self.dynamic = dynamic
        self.dynamic_method = dynamic_method
        self.deficiency = 0
        self.lwc = lwc

        self.rescale = rescale # for channel-rescale
        self.rescale_limit = rescale_limit
        self.lsq = lsq

        self.mode = "calibration"
        self.recorded_quant_input=None
        
        init_value = 4.             # inti value of learnable weight clipping
        if lwc:
            if group_size:
                dim1 = int(shape[0]*math.ceil(shape[1]/group_size))
                self.deficiency = shape[-1]%group_size
                if self.deficiency > 0:
                    self.deficiency = group_size - self.deficiency
                    assert self.symmetric   # support for mlc-llm symmetric quantization
            else:
                dim1 = shape[0]
            self.upbound_factor = nn.Parameter(torch.ones((dim1,1))*init_value)
            self.lowbound_factor = nn.Parameter(torch.ones((dim1,1))*init_value)

        if rescale:
            if rescale_limit:
                self.rescale_param = nn.Parameter(torch.zeros(dim1,1) )
            else:
                self.rescale_param = nn.Parameter(torch.ones(dim1,1) )

        self.sigmoid = nn.Sigmoid()

        self.enable = True
        self.group_size = group_size

    def change_n_bits(self, n_bits):
        self.n_bits = n_bits
        if self.disable_zero_point:
            self.qmin = -(2 ** (n_bits - 1))
            self.qmax = 2 ** (n_bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** (n_bits) - 1

    def quant(self, x, scale, round_zero_point, uint=False):
        x_int = (x / scale).round_()
        if uint:
            x_int = x_int.add_(round_zero_point - self.qmin)
            x_int = x_int.clamp_(0, self.qmax - self.qmin).to(torch.uint8)
            return x_int
        else:
            if not self.symmetric:
                x_int = x_int.add_(round_zero_point)
            x_int = x_int.clamp_(self.qmin, self.qmax).to(torch.int8)
            return x_int

    def dequant(self, x_int, scale, round_zero_point, uint=False):
        if not x_int.dtype == scale.dtype:
            x_int = x_int.to(scale.dtype)
        if uint:
            if round_zero_point is not None:
                x_int = x_int.sub_(round_zero_point - self.qmin)
            x_dequant = x_int.mul_(scale)
            return x_dequant
        else:
            if round_zero_point is not None:
                x_int = x_int.sub_(round_zero_point)
            x_dequant = x_int.mul_(scale)
            return x_dequant

    def fake_quant(self, x, scale, round_zero_point):
        if self.deficiency > 0:
            pad_zeros = torch.zeros((x.shape[0],self.deficiency),dtype=x.dtype,device=x.device)
            x = torch.cat((x,pad_zeros),dim=1)
        
        if self.group_size:
            assert len(x.shape)==2, "only support linear layer now"
            dim1, dim2 = x.shape
            x = x.reshape(-1, self.group_size)
        
        if self.lsq:
            if self.lwc:
                x = x.clone()
                x.data.clamp_(self.xmin_tmp,self.xmax_tmp)

            grad_factor = 1.
            x_dequant = torch._fake_quantize_learnable_per_channel_affine(x,scale.reshape(-1),round_zero_point.reshape(-1),0,self.qmin,self.qmax,grad_factor=grad_factor)
        
        else:
            x_int = round_ste(x / scale)
            if round_zero_point is not None:
                x_int = x_int.add(round_zero_point)
            x_int = x_int.clamp(self.qmin, self.qmax)
            x_dequant = x_int
            if round_zero_point is not None:
                x_dequant = x_dequant.sub(round_zero_point)
            x_dequant = x_dequant.mul(scale)

        if self.group_size:
            x_dequant = x_dequant.reshape(dim1, dim2)
        if self.deficiency > 0:
            x_dequant = x_dequant[:,:-self.deficiency]
        
        if self.rescale:
            rescale_param = self.rescale_param
            if self.rescale_limit:
                rescale_param = 0.5 + F.sigmoid(rescale_param)
            x_dequant = x_dequant*rescale_param

        return x_dequant
    

    def forward(self, x: torch.Tensor):
        if self.n_bits >= 16 or not self.enable:
            return x
        if self.metric == "fix0to1":
            return x.mul_(2**self.n_bits-1).round_().div_(2**self.n_bits-1)

        if self.dynamic_method == "per_token" or self.dynamic_method == "per_channel":
            self.per_token_dynamic_calibration(x)
        elif self.dynamic_method == "per_cluster" or self.mode == "calibration":
            self.calibration(x)  
            return x  
        elif self.dynamic_method == "per_tensor":
            self.per_tensor_calibration(x)

        x_dequant = self.fake_quant(x, self.scale, self.round_zero_point)
        return x_dequant


    def per_token_dynamic_calibration(self, x):
        if self.group_size:
            if self.deficiency == 0:
                x = x.reshape(-1,self.group_size)
            else:
                pad_zeros = torch.zeros((x.shape[0],self.deficiency),dtype=x.dtype,device=x.device)
                x = torch.cat((x,pad_zeros),dim=1)
                x = x.reshape(-1,self.group_size)
        
        if self.dynamic_method == "per_channel":
            if len(self.per_channel_axes):
                assert len(self.per_channel_axes) == 1,"must be one"
                reduce_shape = list(range(x.dim()))
                reduce_shape.remove(self.per_channel_axes[0])
            else:
                reduce_shape = list(range(x.dim()-1))
        else:
            reduce_shape = [-1]

        xmin = x.amin(reduce_shape, keepdim=True)
        xmax =  x.amax(reduce_shape, keepdim=True)
        if self.lwc:
            xmax = self.sigmoid(self.upbound_factor)*xmax
            xmin = self.sigmoid(self.lowbound_factor)*xmin
        if self.symmetric:
            abs_max = torch.max(xmax.abs(),xmin.abs())
            scale = abs_max / (2**(self.n_bits-1)-1)
            self.scale = scale.clamp(min=CLIPMIN, max=1e4)
            zero_point = (2**(self.n_bits-1)-1)*torch.ones_like(self.scale)
        else:
            dynamic_range = xmax - xmin
            scale = dynamic_range / (2**self.n_bits-1)
            self.scale = scale.clamp(min=CLIPMIN, max=1e4)
            zero_point = -(xmin) / (self.scale)
        if self.disable_zero_point:
            self.round_zero_point = None
        else:
            self.round_zero_point = zero_point.clamp(min=-1e4, max=1e4).round()
    
    # RPTQ
    def calibration(self, x: torch.Tensor):
        reduce_axes = [
            _
            for _ in range(x.ndim)
            if _ not in self.per_channel_axes and _ != self.cluster_dim
        ]
        if self.metric in ["layer_mse", "minmax", "ema_minmax"]:
            scale, zero_point = self.minmax_calibration(x, reduce_axes)
        elif self.metric == "mse":
            scale, zero_point = self.mse_calibration(x, reduce_axes)

        del self.scale
        self.register_buffer("scale", scale)
        # self.scale = scale
        if zero_point is not None:
            zero_point.clamp_(min=-1e4, max=1e4)
            del self.zero_point, self.round_zero_point
            self.register_buffer("zero_point", zero_point)
            self.register_buffer("round_zero_point", zero_point.round())

        # debug
        if not self.dynamic:
            if torch.isinf(self.scale).any() or torch.isnan(self.scale).any():
                breakpoint()
            if self.zero_point is not None:
                if (
                    torch.isinf(self.round_zero_point).any()
                    or torch.isnan(self.round_zero_point).any()
                ):
                    breakpoint()
        if x.size(0) == 1:
            self.pack_dim = 1
     
    # RPTQ
    def minmax_calibration(self, x, reduce_axes):
        # minmax
        if self.symmetric:
            if len(reduce_axes):
                abs_max = x.abs().amax(reduce_axes, keepdim=True)
            else:
                abs_max = x.abs()
            scale = abs_max / (2 ** (self.n_bits - 1))
            if self.cluster_dim is not None:
                # for cluster quantization
                st = 0
                for count in self.cluster_counts:
                    part_scale = torch.narrow(scale, self.cluster_dim, st, count)
                    cluster_max = part_scale.amax(
                        self.cluster_dim, keepdim=True
                    )  # 不该保持维度
                    scale.narrow(self.cluster_dim, st, count).copy_(cluster_max)
                    st += count
            scale.clamp_(min=CLIPMIN, max=1e4)
            zero_point = None
        else:
            if len(reduce_axes):
                xmin = x.amin(reduce_axes, keepdim=True)
                xmax = x.amax(reduce_axes, keepdim=True)
            else:
                xmin = x.clone()
                xmax = x.clone()
            if not self.dynamic:
                if self.cached_xmax is not None:
                    if self.metric == "minmax":
                        xmax = torch.max(self.cached_xmax, xmax)
                        xmin = torch.min(self.cached_xmin, xmin)
                    if self.metric == "ema_minmax":
                        xmax = self.cached_xmax * 0.99 + xmax * 0.01
                        xmin = self.cached_xmin * 0.99 + xmin * 0.01
                self.cached_xmax = xmax
                self.cached_xmin = xmin
            if self.cluster_dim is not None:
                # for cluster quantization
                st = 0
                for count in self.cluster_counts:
                    part_xmin = torch.narrow(xmin, self.cluster_dim, st, count)
                    part_xmax = torch.narrow(xmax, self.cluster_dim, st, count)
                    cluster_xmin = part_xmin.amin(self.cluster_dim, keepdim=True)
                    cluster_xmax = part_xmax.amax(self.cluster_dim, keepdim=True)
                    xmin.narrow(self.cluster_dim, st, count).copy_(cluster_xmin)
                    xmax.narrow(self.cluster_dim, st, count).copy_(cluster_xmax)
                    st += count
            scale = (xmax - xmin) * (2**-self.n_bits)
            scale.clamp_(min=CLIPMIN, max=1e4)
            zero_point = (xmax + xmin) * (-0.5 / scale)
            return scale, zero_point

    # RPTQ
    def mse_calibration(self, x, reduce_axes):
        # mse
        if self.symmetric:
            abs_max = x.abs().amax(reduce_axes, keepdim=True)
            min_loss = None
            best_scale = None
            for t in torch.linspace(0.1, 1, steps=50):
                t = abs_max * t
                scale = t / (2 ** (self.n_bits - 1))
                x_sim = self.fake_quant(x, scale, None)
                loss = F.mse_loss(x, x_sim)
                if min_loss is None or min_loss > loss:
                    min_loss = loss
                    best_scale = scale
            scale = best_scale
            zero_point = None
        else:
            xmin = x.amin(reduce_axes, keepdim=True)
            xmax = x.amax(reduce_axes, keepdim=True)
            x_center = (xmin + xmax) / 2
            min_loss = None
            best_scale = None
            best_zero_point = None
            for i in torch.linspace(0, 1, steps=30):
                t_left = xmin * i + x_center * (1 - i)
                for j in torch.linspace(0.1, 1, steps=30):
                    t_right = t_left * j + xmax * (1 - j)
                    if (t_left == t_right).any():
                        continue
                    scale = (t_right - t_left) / (2**self.n_bits)
                    zero_point = -((t_right + t_left) / 2 / scale)
                    x_sim = self.fake_quant(x, scale, zero_point)
                    loss = F.mse_loss(x, x_sim, reduce=False).mean(
                        reduce_axes, keepdim=True
                    )
                    if min_loss is None:
                        min_loss = loss
                        best_scale = scale
                        best_zero_point = zero_point
                    else:
                        mask = min_loss > loss
                        best_scale[mask] = scale
                        best_zero_point[mask] = zero_point
            scale = best_scale
            zero_point = best_zero_point
        scale.clamp_(min=1e-6, max=1e4)
        return scale, zero_point

    # I-LLM
    def per_tensor_calibration(self,x):
        xmin = x.min()
        xmax = x.max()
        if self.symmetric or self.disable_zero_point:
            abs_max = torch.max(xmax.abs(), xmin.abs())
            scale = abs_max / (2 ** (self.n_bits - 1) - 1)
            self.scale = scale.clamp(min=CLIPMIN, max=1e4)
            self.round_zero_point = None
        else:
            range = xmax - xmin
            scale = range / (2**self.n_bits - 1)
            self.scale = scale.clamp(min=CLIPMIN, max=1e4)
            zero_point = -(xmin) / (self.scale)
            self.round_zero_point = zero_point.clamp(min=-1e4, max=1e4).round()

    def layer_mse_param_search_update(
        self, qlayer, a_quantizers, inps, outs, attention_mask, steps=10
    ):
        assert not self.dynamic and (
            not self.symmetric
        ), "only support static asymmetric in layer_mse_calibration now"

        # prepare_layer_mse_calibration
        self.qlayer = qlayer
        self.attention_mask = attention_mask
        self.quantizers_nbit_dict = {}
        if self.n_bits == 16:
            return
        for name, quantizer in a_quantizers.items():
            self.quantizers_nbit_dict[quantizer] = quantizer.n_bits
            quantizer.n_bits = 16

        self.n_bits = self.quantizers_nbit_dict[self]
        xmax = self.cached_xmax
        xmin = self.cached_xmin
        xrange = xmax - xmin
        step_interval = xrange / steps

        print("=== start layer_mse_param_search_update ===")
        dev = self.scale.device
        attention_mask = self.attention_mask.to(dev)

        nsamples = inps.size(0)
        best_mse = 0
        for j in range(nsamples):
            sim_outs = qlayer(
                inps[j].unsqueeze(0).to(dev), attention_mask=attention_mask
            )[0]
            best_mse += F.mse_loss(sim_outs[0], outs[j].to(dev))
        print(f"best_mse {best_mse}")

        if self.cluster_dim is not None:
            # for cluster quantization
            st = 0
            for count in self.cluster_counts:
                part_xmin = torch.narrow(xmin, self.cluster_dim, st, count)
                part_xmax = torch.narrow(xmax, self.cluster_dim, st, count)
                part_step_interval = torch.narrow(
                    step_interval, self.cluster_dim, st, count
                )
                for i in range(steps):
                    # left
                    new_part_xmin = part_xmin + part_step_interval
                    new_part_scale = (part_xmax - new_part_xmin) * (2**-self.n_bits)
                    new_part_scale.clamp_(min=CLIPMIN, max=1e4)
                    self.scale.narrow(self.cluster_dim, st, count).copy_(new_part_scale)
                    new_part_zero_point = (part_xmax + new_part_xmin) * (
                        -0.5 / new_part_scale
                    )
                    new_part_zero_point.clamp_(min=-1e4, max=1e4).round_()
                    self.round_zero_point.narrow(self.cluster_dim, st, count).copy_(
                        new_part_zero_point
                    )
                    mse = 0
                    for j in range(nsamples):
                        sim_outs = qlayer(
                            inps[j].unsqueeze(0).to(dev), attention_mask=attention_mask
                        )[0]
                        mse += F.mse_loss(sim_outs[0], outs[j].to(dev))
                    print(f"mse {mse}")
                    if mse < best_mse:
                        # accept
                        best_mse = mse
                        part_xmin = new_part_xmin
                        print(f"update min, {best_mse}")

                    # right
                    new_part_xmax = part_xmax - part_step_interval
                    new_part_scale = (new_part_xmax - part_xmin) * (2**-self.n_bits)
                    new_part_scale.clamp_(min=CLIPMIN, max=1e4)
                    self.scale.narrow(self.cluster_dim, st, count).copy_(new_part_scale)
                    new_part_zero_point = (new_part_xmax + part_xmin) * (
                        -0.5 / new_part_scale
                    )
                    new_part_zero_point.clamp_(min=-1e4, max=1e4).round_()
                    self.round_zero_point.narrow(self.cluster_dim, st, count).copy_(
                        new_part_zero_point
                    )
                    mse = 0
                    for j in range(nsamples):
                        sim_outs = qlayer(
                            inps[j].unsqueeze(0).to(dev), attention_mask=attention_mask
                        )[0]
                        mse += F.mse_loss(sim_outs[0], outs[j].to(dev))
                    if mse < best_mse:
                        # accept
                        best_mse = mse
                        part_xmax = new_part_xmax
                        print(f"update max, {best_mse}")
                st += count

        # free_layer_mse_calibration
        self.qlayer = None
        self.attention_mask = None
        for quantizer, n_bits in self.quantizers_nbit_dict.items():
            quantizer.n_bits = n_bits
        self.quantizers_nbit_dict = None

    # I-LLM
    def normal_quantize(self, x, scales: torch.Tensor, mig_cof: torch.Tensor):
        s = (scales / mig_cof).max()
        s = s / (2**self.n_bits - 1)
        self.scale = s
        # only support symmetric quantization
        self.round_zero_point = None
        
    def scale_frexp(self):
        k = 16
        m = (self.scale*(2**k)).round()
        self.scale = m*(2**(-k))
        
        return self.scale

    def quant2int(self, x):
        if self.n_bits >= 16 or not self.enable:
            return x
        if self.metric == "fix0to1":
            return x.mul_(2**self.n_bits - 1).round_().div_(2**self.n_bits - 1)

        if self.deficiency > 0:
            pad_zeros = torch.zeros(
                (x.shape[0], self.deficiency), dtype=x.dtype, device=x.device
            )
            x = torch.cat((x, pad_zeros), dim=1)

        if self.group_size:
            assert len(x.shape) == 2, "only support linear layer now"
            dim1, dim2 = x.shape
            x = x.reshape(-1, self.group_size)
        x_int = round_ste(x / self.scale)
        if self.round_zero_point is not None:
            x_int = x_int.add(self.round_zero_point)
        x_int = x_int.clamp(self.qmin, self.qmax)
        
        if self.group_size:
            x_int = x_

    def illm_dequant(self, x_int):
        if self.group_size:
            assert len(x_int.shape) == 2, "only support linear layer now"
            dim1, dim2 = x_int.shape
            x_int = x_int.reshape(-1, self.group_size)
            
        x_dequant = x_int
        if self.round_zero_point is not None:
            x_dequant = x_dequant.sub(self.round_zero_point)
        x_dequant = x_dequant.mul(self.scale)
        if self.group_size:
            x_dequant = x_dequant.reshape(dim1, dim2)
        if self.deficiency > 0:
            x_dequant = x_dequant[:, : -self.deficiency]

        if self.rescale:
            rescale_param = self.rescale_param
            if self.rescale_limit:
                rescale_param = F.sigmoid(rescale_param) + 0.5
            x_dequant = x_dequant*self.rescale_param
        return x_dequant
    

    def register_scales_and_zeros(self):
        self.register_buffer('scales', self.scale)
        self.register_buffer('zeros', self.round_zero_point)
        del self.scale
        del self.round_zero_point
