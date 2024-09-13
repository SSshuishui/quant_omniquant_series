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

from models.int_llama_layer import QuantLlamaDecoderLayer
from models.int_opt_layer import QuantOPTDecoderLayer
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


@torch.no_grad()
def evaluate(lm, args, logger):
    results = {}
    # if "opt" in args.net.lower():
    #     lm.model.model.decoder = lm.model.model.decoder.to(lm.device)
    # elif "llama" in args.net.lower() or "mixtral" in args.net.lower():
    #     lm.model = lm.model.to(lm.device)
    # elif "falcon" in args.net.lower():
    #     lm.model.transformer = lm.model.transformer.to(lm.device)

    hf_device_map = lm.model.hf_device_map
    hf_device = f"cuda:{hf_device_map[f'model.layers.{len(lm.model.model.layers)-1}']}"

    if args.eval_ppl:
        for dataset in ["wikitext2", "c4", "ptb"]:
            cache_testloader = f'{args.cache_dir}/testloader_{args.model_family}_{dataset}_all.cache'
            if os.path.exists(cache_testloader):
                testloader = torch.load(cache_testloader)
                logger.info(f"load calibration from {cache_testloader}")
            else:
                dataloader, testloader = get_loaders(
                    dataset,
                    seed=args.seed,
                    model=args.model,
                    seqlen=lm.seqlen,
                )
                torch.save(testloader, cache_testloader)
            if "c4" in dataset:
                testenc = testloader
            else:
                testenc = testloader.input_ids

            nsamples = testenc.numel() // lm.seqlen
            use_cache = lm.model.config.use_cache
            lm.model.config.use_cache = False
            lm.model.eval()
            nlls = []
            for i in tqdm(range(nsamples)):
                batch = testenc[:, (i * lm.seqlen) : ((i + 1) * lm.seqlen)].to(hf_device)
                if "opt" in args.net.lower():
                    logs = lm.model.model.decoder(batch)
                elif "llama" in args.net.lower() or "mixtral" in args.net.lower():
                    logs = lm.model.model(batch)
                elif "falcon" in args.model:
                    logs = lm.model.transformer(batch)
                hidden_states = logs[0]
                logits = lm.model.lm_head(hidden_states)
                shift_logits = logits[:, :-1, :]
                shift_labels = testenc[:, (i * lm.seqlen) : ((i + 1) * lm.seqlen)][:, 1:].to(lm.model.lm_head.weight.device)
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                neg_log_likelihood = loss.float() * lm.seqlen
                nlls.append(neg_log_likelihood)
                if i == args.limit:
                    break

            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * lm.seqlen))
            logger.info(f'{dataset} : {ppl.item()}')
            lm.model.config.use_cache = use_cache
            results[dataset] = ppl.item()
            
    if args.tasks != "":
        t_results = evaluator.simple_evaluate(
            lm,
            tasks=args.tasks,
            num_fewshot=args.num_fewshot,
            limit=None if args.limit == -1 else args.limit,
        )
        results.update(t_results)
        logger.info(results)
        pprint(results)
        # for test of MMLU
        if 'hendrycksTest' in args.tasks:
            all_cors = []
            all_cors_norm = []
            subcat_cors = {subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists}
            cat_cors = {cat: [] for cat in categories}
            cat_cors_norm = {cat: [] for cat in categories}
            for key in t_results['results'].keys():
                if not 'hendrycksTest' in key:
                    continue
                subject = key.split('-')[-1]
                cors = t_results['results'][key]['acc']
                cors_norm = t_results['results'][key]['acc_norm']
                subcats = subcategories[subject]
                for subcat in subcats:
                    subcat_cors[subcat].append(cors)
                    for key in categories.keys():
                        if subcat in categories[key]:
                            cat_cors[key].append(cors)
                            cat_cors_norm[key].append(cors_norm)
                    all_cors.append(cors)
                    all_cors_norm.append(cors_norm)
                    
            for cat in cat_cors:
                cat_acc = np.mean(cat_cors[cat])
                logger.info("Average accuracy {:.4f} - {}".format(cat_acc, cat))
            weighted_acc = np.mean(all_cors)
            logger.info("Average accuracy: {:.4f}".format(weighted_acc))               
    return results


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model name of model path")
    parser.add_argument("--method", type=str, help="quantization method")
    parser.add_argument("--cache_dir", default="./cache", type=str, help="cache dir of dataset, leading to faster debug")
    parser.add_argument("--log_dir", default="./log/", type=str, help="direction of logging file")
    parser.add_argument("--save_dir", default=None, type=str, help="direction for saving parameters and fake quantization model")
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
    logger = utils.create_logger(args.net, log_dir)
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
        "disable_zero_point": args.disable_zero_point
    }
    args.act_quant_params = {
        "n_bits":  args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.q_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.k_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.v_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
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
    elif args.method == "affinequant":
        from quantize.affinequant import affinequant

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
            if isinstance(module,QuantLlamaDecoderLayer) or isinstance(module,QuantOPTDecoderLayer):
                if args.let:
                    del module.qkv_smooth_scale
                    del module.qkv_smooth_shift
                    del module.out_smooth_scale
                    del module.out_smooth_shift
                    del module.fc1_smooth_scale
                    del module.fc1_smooth_shift           
        lm.model.save_pretrained(args.save_dir)  
        lm.tokenizer.save_pretrained(args.save_dir) 

    logger.info("=== start evaluation ===")
    evaluate(lm, args,logger)


if __name__ == "__main__":
    print(sys.argv)
    main()
