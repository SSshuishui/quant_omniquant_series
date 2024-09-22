import os
import sys
import random
import numpy as np
from models.LMClass import LMClass
import torch
import time
from datautils import get_loaders
from lm_eval import evaluator
from pprint import pprint
import torch.nn as nn
from tqdm import tqdm
import utils
from pathlib import Path
from categories import subcategories, categories

from quantize.int_linear import QuantLinear

import pdb


torch.backends.cudnn.benchmark = True

net_choices = [
    "opt-125m",
    "opt-1.3b",
    "opt-2.7b",
    "opt-6.7b",
    "opt-13b",
    "opt-30b",
    "opt-66b",
    "llama-7b",
    "llama-13b",
    "llama-30b",
    "llama-65b",
    "Llama-2-7b",
    "Llama-2-13b",
    "Llama-2-70b",
    "Llama-2-7b-chat",
    "Llama-2-13b-chat",
    "Llama-3-8b",
    "Llama-3.1-8b",
    "llava-llama-2-13b-chat-lightning-preview",
    "falcon-180b",
    "falcon-7b",
    "mixtral-8x7b"]


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model name of model path")
    parser.add_argument("--method", type=str, help="quantization method")
    parser.add_argument("--cache_dir", default="./cache", type=str, help="cache dir of dataset, leading to faster debug")
    parser.add_argument("--log_dir", default="./log/", type=str, help="direction of logging file")
    parser.add_argument("--save_dir", default='./save_pth', type=str, help="direction for saving parameters and fake quantization model")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--real_quant", default=False, action="store_true", help="real quantization, which can see memory reduce. Note that due to the limitations of AutoGPTQ kernels, the real quantization of weight-only quantization can only lead memory reduction, but with slower inference speed.")
    parser.add_argument("--calib_dataset",type=str,default="wikitext2",
        choices=["wikitext2", "ptb", "c4", "mix", "pile"],
        help="Where to extract calibration data from.")
    parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration data samples.")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size.")
    parser.add_argument("--seed", type=int, default=2, help="Seed for sampling the calibration data.")
    parser.add_argument("--tasks", default="")
    parser.add_argument("--eval_ppl", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--wbits", type=int, default=4)
    parser.add_argument("--abits", type=int, default=16)
    parser.add_argument("--group_size", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--let_lr", type=float, default=5e-3)
    parser.add_argument("--lwc_lr", type=float, default=1e-2)
    parser.add_argument("--wd", type=float, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--let",default=False, action="store_true",help="activate learnable equivalent transformation")
    parser.add_argument("--lwc",default=False, action="store_true",help="activate learnable weight clipping")
    parser.add_argument("--aug_loss", default=False, action="store_true", help="calculate additional loss with same input")
    parser.add_argument("--symmetric",default=False, action="store_true", help="symmetric quantization")
    parser.add_argument("--disable_zero_point",default=False, action="store_true", help="quantization without zero_point")
    parser.add_argument("--a_dynamic_method", type=str, default="per_token", choices=["per_token"])
    parser.add_argument("--w_dynamic_method", type=str, default="per_channel", choices=["per_channel"])
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--deactive_amp", action="store_true", help="deactivate AMP when 8<=bits<16")
    parser.add_argument("--net", type=str, default=None, choices=net_choices)
    parser.add_argument("--act-scales", type=str, default=None)
    parser.add_argument("--act-shifts", type=str, default=None)

    # For Omniquant Args
    parser.add_argument(
        "--attn_implementation",
        type=str, required=False, default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="attention implementation that the model works with",
    )
    
    # For AffineQuant Args
    parser.add_argument("--use_matrix", default=False, action="store_true", help="qkt affine mateix or not")
    parser.add_argument("--use_ln_matrix",default=False, action="store_true", help="layernorm vector or matrix")
    parser.add_argument('--sf',"--stability_factor",type=float, default=1.0, help="stability factor for gradual mask")

    # For RPTQ Args
    parser.add_argument(
        "--metric", type=str, default="ema_minmax", choices=["minmax", "ema_minmax", "mse", "layer_mse"],
    )
    parser.add_argument("--disable_w_quant", action="store_true")
    parser.add_argument("--disable_a_quant", action="store_true")
    parser.add_argument("--R1_clusters", type=int, default=32)
    parser.add_argument("--R2_clusters", type=int, default=4)
    parser.add_argument("--R3_clusters", type=int, default=4)
    parser.add_argument("--R4_clusters", type=int, default=32)
    parser.add_argument("--R5_clusters", type=int, default=32)
    parser.add_argument("--reorder", type=str, default="12345", help="like 12345 or 1")
    parser.add_argument("--w_quantizer", type=str, default="gptq", choices=["gptq", "normal"])
    parser.add_argument("--only_quant_kv", action="store_true", help="only quantize the kv cache")


    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # check
    if args.epochs > 0:
        assert args.lwc or args.let
        
    if (args.wbits<16 and args.wbits>=8) or (args.abits<16 and args.abits>=8):
        args.deactive_amp = True

    # set net and model_family
    if args.net is None:
        args.net = args.model.split('/')[-1]  
    args.model_family = args.net.split('-')[0]

    # init logger
    if args.log_dir:
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    if args.cache_dir:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    logger = utils.create_logger(args.method, log_dir)
    logger.info(args)
    
    # load model
    lm = LMClass(args)
    lm.seqlen = 2048
    lm.model.eval()
    for param in lm.model.parameters():
        param.requires_grad = False

    args.weight_quant_params = {
        "n_bits": args.wbits,
        "per_channel_axes": [0],
        "symmetric": args.symmetric,
        "dynamic_method": args.w_dynamic_method,
        "group_size": args.group_size,
        "lwc":args.lwc,
        "disable_zero_point": args.disable_zero_point,
        "metric": "minimax"
    }
    args.act_quant_params = {
        "n_bits":  16 if args.only_quant_kv else args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
        "metric": args.metric
    }
    args.q_quant_params = {
        "n_bits": 16 if args.only_quant_kv else args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
        "metric": args.metric
    }
    args.k_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
        "metric": args.metric
    }
    args.v_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
        "metric": args.metric
    }
    args.p_quant_params = {
        "n_bits": 16,
        "metric": "fix0to1",
    }

    # act scales and shifts
    if args.act_scales is None:
        args.act_scales = f'./act_scales/{args.net}.pt'
    if args.act_shifts is None:
        args.act_shifts = f'./act_shifts/{args.net}.pt'
    

    logger.info("=== start quantization ===")
    # load calibration dataset
    cache_dataloader = f'{args.cache_dir}/dataloader_{args.model_family}_{args.calib_dataset}_{args.nsamples}.cache'
    if os.path.exists(cache_dataloader):
        print("load calibration from cache")
        dataloader = torch.load(cache_dataloader)
        logger.info(f"load calibration from {cache_dataloader}")
    else:
        print("load calibration from dataloader")
        dataloader, _ = get_loaders(
            args.calib_dataset,
            nsamples=args.nsamples,
            seed=args.seed,
            model=args.model,
            seqlen=lm.seqlen,
        )
        torch.save(dataloader, cache_dataloader)
    act_scales = None
    act_shifts = None
    if args.let:
        act_scales = torch.load(args.act_scales)
        act_shifts = torch.load(args.act_shifts)

    # quantization
    if args.method == "omniquant" and args.wbits < 16 or args.abits <16:
        from quantize.omniquant import omniquant
        from models.int_llama_layer import OmniQuantLlamaDecoderLayer
        from models.int_opt_layer import OmniQuantOPTDecoderLayer
        from eval_ppl_utils import evaluate

        tick = time.time()     
        omniquant(
            lm,
            args,
            dataloader,
            act_scales,
            act_shifts,
            logger,
        )
        logger.info(time.time() - tick)

        if args.save_dir:
            logger.info("=== save model ===")
            # delete omni parameters
            for name, module in lm.model.named_modules():
                if isinstance(module, QuantLinear):
                    del module.weight_quantizer.lowbound_factor
                    del module.weight_quantizer.upbound_factor
                if isinstance(module,OmniQuantLlamaDecoderLayer) or isinstance(module,OmniQuantOPTDecoderLayer):
                    if args.let:
                        del module.qkv_smooth_scale
                        del module.qkv_smooth_shift
                        del module.out_smooth_scale
                        del module.out_smooth_shift
                        del module.fc1_smooth_scale
                        del module.fc1_smooth_shift           
            lm.model.save_pretrained(args.save_dir+"/omniquant")  
            lm.tokenizer.save_pretrained(args.save_dir+"/omniquant") 
        
        logger.info("=== start evaluation ===")
        evaluate(lm, args,logger)

    elif args.method == "affinequant":
        from quantize.affinequant import affinequant
        from models.int_llama_layer import AffineQuantLlamaDecoderLayer
        from models.int_opt_layer import AffineQuantOPTDecoderLayer
        from eval_ppl_utils import evaluate

        tick = time.time()
        affinequant(
            lm, 
            args, 
            dataloader, 
            act_scales, 
            act_shifts, 
            logger
        )   
        logger.info(time.time() - tick)

        if args.save_dir:
            logger.info("=== save model ===")
            # delete omni parameters
            for name, module in lm.model.named_modules():
                if isinstance(module, QuantLinear):
                    del module.weight_quantizer.lowbound_factor
                    del module.weight_quantizer.upbound_factor
                if isinstance(module,AffineQuantLlamaDecoderLayer) or isinstance(module,AffineQuantOPTDecoderLayer):
                    if args.let:
                        del module.qkv_smooth_scale
                        del module.qkv_smooth_shift
                        del module.out_smooth_scale
                        del module.out_smooth_shift
                        del module.fc1_smooth_scale
                        del module.fc1_smooth_shift           
            lm.model.save_pretrained(args.save_dir+"/affinequant")  
            lm.tokenizer.save_pretrained(args.save_dir+"/affinequant") 

        logger.info("=== start evaluation ===")
        evaluate(lm, args,logger)

    elif args.method == "lrquant" and args.wbits < 16 or args.abits <16:
        from quantize.lrquant import lrquant
        from models.int_llama_layer import LRQuantLlamaDecoderLayer
        from models.int_opt_layer import LRQuantOPTDecoderLayer
        from eval_ppl_utils import evaluate_lrquant
        
        fp_lm = copy.deepcopy(lm)
        fp_lm.model.eval()
        for fp_param in fp_lm.model.parameters():
            fp_param.requires_grad = False
        
        tick = time.time()
        lrquant(
            lm, 
            args, 
            dataloader, 
            act_scales, 
            act_shifts, 
            logger
        )   
        logger.info(time.time() - tick)

        if args.save_dir:
            logger.info("=== save model ===")
            # delete omni parameters
            for name, module in lm.model.named_modules():
                if isinstance(module, QuantLinear):
                    del module.weight_quantizer.lowbound_factor
                    del module.weight_quantizer.upbound_factor

                if isinstance(module,LRQuantLlamaDecoderLayer) or isinstance(module,LRQuantOPTDecoderLayer):
                    if args.let:
                        del module.qkv_smooth_scale
                        del module.qkv_smooth_shift
                        del module.out_smooth_scale
                        del module.out_smooth_shift
                        del module.fc1_smooth_scale
                        del module.fc1_smooth_shift           
            lm.model.save_pretrained(args.save_dir+"/lrquant")  
            lm.tokenizer.save_pretrained(args.save_dir+"/lrquant") 

        logger.info("=== start evaluation ===")
        evaluate_lrquant(lm, args,logger, fp_lm)

    elif args.method == "rptq":
        from quantize.reorderquant import rptq
        from models.int_llama_layer import RPTQLlamaDecoderLayer
        from models.int_opt_layer import RPTQOPTDecoderLayer
        from eval_ppl_utils import evaluate_rptq

        args.layer_norm_out_quant_params = {
            "n_bits": 16 if args.only_quant_kv else max(8, args.abits),
            "per_channel_axes": [],
            "symmetric": False,
            "metric": args.metric,
            "dynamic": args.a_dynamic,
        }
        n_clusters = {
            "R1": args.R1_clusters,
            "R2": args.R2_clusters,
            "R3": args.R3_clusters,
            "R4": args.R4_clusters,
            "R5": args.R5_clusters,
        }
        tick = time.time()
        rptq( 
            lm,
            args,
            dataloader,
            n_clusters,
            args.reorder,
            logger
        )
        for layer in lm.model.model.decoder.layers:
            if hasattr(layer, "set_quant_state"):
                layer.set_quant_state(
                    not args.disable_w_quant, not args.disable_a_quant
                )
        logger.info(time.time() - tick)
    


if __name__ == "__main__":
    print(sys.argv)
    main()
