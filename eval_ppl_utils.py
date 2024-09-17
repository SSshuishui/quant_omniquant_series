from datautils import get_loaders
from lm_eval import evaluator
from pprint import pprint
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import os
import copy
from categories import subcategories, categories
import pdb



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
    hf_device = f"cuda:{hf_device_map[f'model.layers.{0}']}"
    print("hf_device_map: ", hf_device_map, "hf_device: ", hf_device)
    
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


@torch.no_grad()
def evaluate_lrquant(lm, args, logger, fp_lm):
    results = {}
    # torch.save(lm.model.state_dict(),os.path.join(args.output_dir, f"current.pth"))
    if args.multigpu:
        if "opt" in args.net.lower():
            map_layers_to_multi_gpus(lm.model.model.decoder.layers)
            input_device = lm.model.model.decoder.layers[0].device
            output_device = lm.model.model.decoder.layers[-1].device
            lm._device = input_device
            assert input_device == output_device
            lm.model.model.decoder.embed_positions.to(input_device)
            lm.model.model.decoder.embed_tokens.to(input_device)
            lm.model.model.decoder.final_layer_norm.to(output_device)
            lm.model.lm_head.to(output_device)

        elif "llama" in args.net.lower():
            map_layers_to_multi_gpus(lm.model.model.layers)
            input_device = lm.model.model.layers[0].device
            output_device = lm.model.model.layers[-1].device
            assert input_device == output_device
            lm._device = input_device
            lm.model.model.embed_tokens.to(input_device)
            lm.model.model.norm.to(output_device)
            lm.model.lm_head.to(output_device)
        elif "falcon" in args.net.lower():
            map_layers_to_multi_gpus(lm.model.transformer.h)
            input_device = lm.model.transformer.h[0].device
            output_device = lm.model.transformer.h[-1].device
            assert input_device == output_device
            lm._device = input_device
            lm.model.transformer.word_embeddings.to(input_device)
            lm.model.transformer.ln_f.to(output_device)
            lm.model.lm_head.to(output_device)
    else:
        if "opt" in args.net.lower():
            lm.model.model.decoder = lm.model.model.decoder.to(lm.device)
        elif "llama" in args.net.lower():
            lm.model = lm.model.to(lm.device)
        elif "falcon" in args.net.lower():
            lm.model.transformer = lm.model.transformer.to(lm.device)

    if args.eval_ppl:
        for dataset in ["wikitext2","ptb","c4","ptb-new",'c4-new']:
            cache_testloader = f'{args.cache_dir}/testloader_{args.model_family}_{dataset}_all.cache'                   

            if os.path.exists(cache_testloader):
                testloader = torch.load(cache_testloader)
                logger.info(f"load calibration from {cache_testloader}")
            else:
                _, testloader = get_loaders(
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

            lm.model.load_state_dict(torch.load(os.path.join(args.output_dir, f"current.pth")))
            lm.model.eval()

            nsamples = testenc.numel() // lm.seqlen
            use_cache = lm.model.config.use_cache
            lm.model.config.use_cache = False

            fp_lm.model.config.use_cache = False
            fp_lm.model.eval()

            lm2 = copy.deepcopy(lm)
            lm2.model = lm2.model.cpu()
            lm2.model.config.use_cache = False
            lm2.model.eval()

            if dataset != args.calib_dataset and args.tta:   
                lm.model = lm.model.cpu()     
                lm2.model = lm2.model.to(lm2.device)
                lm2.model.eval()
                tta_loader = []
                for i in range(nsamples):
                    tta_loader.append(testenc[:, (i * lm.seqlen) : ((i + 1) * lm.seqlen)])
                with torch.enable_grad():#torch.enable_grad():
                    tta(
                        lm2,
                        args,
                        tta_loader, #dataloader_test
                        fp_lm,
                        logger                        
                    )
                lm2.model = lm2.model.to(lm.device)
                lm2.model.eval()
                tmp_lm = lm2
            else:
                lm.model = lm.model.to(lm.device)
                tmp_lm = lm

            nlls = []
            for i in tqdm(range(nsamples)):
                batch = testenc[:, (i * tmp_lm.seqlen) : ((i + 1) * tmp_lm.seqlen)].to(tmp_lm.device) #1*2048  
                #c4 testenc:524288 = 2048*256
                # x = batch[0] #2048
                if "opt" in args.net.lower():
                    outputs = tmp_lm.model.model.decoder(batch)
                elif "llama" in args.net.lower():
                    outputs = tmp_lm.model.model(batch) # 1*2048*4096
                    hidden_states = outputs[0] #1*2048*4096
                    logits = tmp_lm.model.lm_head(hidden_states) #1*2048*32000
                shift_logits = logits[:, :-1, :] #1*2047*32000
                shift_labels = testenc[:, (i * tmp_lm.seqlen) : ((i + 1) * tmp_lm.seqlen)][
                    :, 1:
                ].to(tmp_lm.model.lm_head.weight.device) #1*2047
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                neg_log_likelihood = loss.float() * tmp_lm.seqlen
                nlls.append(neg_log_likelihood)
                if i == args.limit:
                    break                                        
            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * tmp_lm.seqlen))
            logger.info(f'{dataset} : {ppl.item()}')
            tmp_lm.model.config.use_cache = use_cache
            results[dataset] = ppl.item()
            tmp_lm.model = tmp_lm.model.cpu()
            tmp_lm.model.config.use_cache = False
            tmp_lm.model.eval()        
    
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
