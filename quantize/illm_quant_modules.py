import torch
import torch.nn as nn
import torch.nn.functional as F
from quantize.quantizer import UniformAffineQuantizer


# C8C8Add
class QuantAdd(nn.Module):
    def __init__(self,
                 x1_quant_params: dict = {},
                 x2_quant_params: dict = {},
                 ):
        super().__init__()
        self.x1_quantizer = UniformAffineQuantizer(**x1_quant_params)
        self.x2_quantizer = UniformAffineQuantizer(**x2_quant_params)
        self.use_act_quant = False
    
    def forward(self,x1,x2):
        if self.use_act_quant:
            x1 = self.x1_quantizer(x1)
            x2 = self.x2_quantizer(x2)
        return x1 + x2
    

class QuantSoftmax(nn.Module):
    def __init__(self,act_quant_params:dict = dict(),dim=-1):
        super().__init__()
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)
        self.dim = dim
        self.use_act_quant = False
    
    def forward(self,attn_weights,attention_mask=None):
        ret_dtype = attn_weights.dtype
        if self.use_act_quant:
            attn_weights = self.act_quantizer(attn_weights)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
        return F.softmax(attn_weights,dim=-1,dtype=torch.float32).to(ret_dtype)

class QuantSwiglu(nn.Module):
    def __init__(self,x1_quant_params=dict(),x2_quant_params = dict()):
        super().__init__()
        self.x1_quantizer = UniformAffineQuantizer(**x1_quant_params)
        self.x2_quantizer = UniformAffineQuantizer(**x2_quant_params)
        self.smooth = None #  smooth factor
        self.use_act_quant = False

    def forward(self,x1,x2):
        if self.use_act_quant:
            x1 = self.x1_quantizer(x1)
            x2 = self.x2_quantizer(x2)
        if self.smooth is  None:
            return x1 *  F.sigmoid(x1) * x2
        else:
            return x1 * F.sigmoid(x1 / self.smooth.to(x1.device)) * x2
        
class FSBRLayerNorm(nn.Module):
    def __init__(self, ori_layer_norm,act_quant_params=dict(n_bits=8,symmetric=False,per_channel_axes=[2])) -> None:
        super().__init__()
        self.register_buffer("weight", ori_layer_norm.weight)
        if ori_layer_norm.bias is not None:
            self.register_buffer("bias", ori_layer_norm.bias)
        else:
            self.bias = None
        self.eps = ori_layer_norm.eps
        self.norm_func = nn.functional.layer_norm
        self.normalized_shape = ori_layer_norm.normalized_shape
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)
        self.use_act_quant = False
        self.use_temporary_parameter = False

    def forward(self, x):
        if self.use_act_quant:
            x = self.act_quantizer(x)
        if self.use_temporary_parameter:
            weight = self.temp_weight
            bias = self.temp_bias
        else:
            weight = self.weight
            bias = self.bias
        out = self.norm_func(x, self.normalized_shape, weight, bias, eps=self.eps)
        return out

    def set_quant_state(self, use_weight_quant, use_act_quant):
        self.use_act_quant = use_act_quant


class FSBRLlamaRMSNorm(nn.Module):
    def __init__(
        self,
        ori_norm,
        act_quant_params: dict = dict(
            n_bits=8,
            symmetric=False,
            per_channel_axes=[2],
        ),
        eps=1e-6,
    ):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.register_buffer("weight", ori_norm.weight)
        self.bias = None
        self.variance_epsilon = eps
        self.use_temporary_parameter = False
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)
        self.use_act_quant = False

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        if self.use_act_quant:
            hidden_states = self.act_quantizer(hidden_states)
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        if self.use_temporary_parameter:
            weight = self.temp_weight
            bias = self.temp_bias
        else:
            weight = self.weight
            bias = self.bias if hasattr(self, "bias") else None

        return (
            (weight * hidden_states + bias).to(input_dtype)
            if bias is not None
            else (weight * hidden_states).to(input_dtype)
        )
    
