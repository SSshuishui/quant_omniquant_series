
Include:
| Methods | Quantize | PPL Eval | Task Eval | Save |
| :--- | ---: | :---: | :---: | :---: 
| OmniQuant | ✅ | ✅ | TODO | ✅ 
| AffineQuant | ✅ | ✅ | TODO | TODO 
| LRQuant | TODO | TODO | TODO | TODO 
| RPTQ | TODO | TODO | TODO | TODO 
| Slim-Plus | TODO | TODO | TODO | TODO 
| I-LLM | TODO | TODO | TODO | TODO 

## Install
```
conda create -n quant_omniquant python=3.10 -y
conda activate quant_omniquant
git clone https://github.com/SSshuishui/quant_omniquant_series.git
cd quant_omniquant_series
pip install --upgrade pip 
pip install -e .
```

We also leverage the kernel from [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ) to achieve real quantization. So you should also install the bug-fixed AutoGPTQ as follows::
```
git clone https://github.com/ChenMnZ/AutoGPTQ-bugfix
pip install -v .
```

## Usage
**We provide full script to run OmniQuant in `./scripts/`**. We use LLaMa-8B as an example here:
1. Obtain the channel-wise scales and shifts required for initialization:
you can generate channel-wise scales and shifts by yourself:
```
python generate_act_scale_shift.py --model /PATH/TO/LLaMA3/
```

### For OmniQuant
2. Weight-only quantization
```
# W3A16
python main.py \
--method omniquant \
--model /PATH/TO/LLaMA3/  \
--epochs 20 --log_dir ./log/llama3-8b-w3a16 \
--eval_ppl --wbits 3 --abits 16 --lwc

# W3A16g128
python main.py \
--method omniquant \
--model /PATH/TO/LLaMA3/  \
--epochs 20 --log_dir ./log/llama3-8b-w3a16g128 \
--eval_ppl --wbits 3 --abits 16 --group_size 128 --lwc
```

3. weight-activation quantization
```
# W4A4
python main.py \
--method omniquant \
--model /PATH/TO/LLaMA3/  \
--epochs 20 --log_dir ./log/llama3-8b-w4a4 \
--eval_ppl --wbits 4 --abits 4 --lwc --let \
--tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande
```

4. evaluation
take LLaMa-7B with W3A16g128 quantization as an example:
```
python main.py \
--method omniquant \
--model /PATH/TO/LLaMA3/  \
--epochs 0 --log_dir ./log/test \
--eval_ppl --wbits 3 --abits 16 --group_size 128 --lwc \
--resume /PATH/TO/Pretrained/Parameters 
```

### For AffineQuant
2. Weight-only quantization
```
# W3A16
python main.py \
--method affinequant \
--model /PATH/TO/LLaMA3 \
--epochs 20 --log_dir ./log/llama3-8b-w3a16 \
--eval_ppl --wbits 3 --abits 16 --lwc --use_ln_matrix --sf 1e-2

# W3A16g128
python main.py \
--method affinequant \
--model /PATH/TO/LLaMA3/llama3-8b  \
--epochs 20 --log_dir ./log/llama3-8b-w3a16g128 \
--eval_ppl --wbits 3 --abits 16 --group_size 128 --lwc --use_ln_matrix --sf 1e-2
```

3. weight-activation quantization
```
# W4A4
python main.py \
--method affinequant \
--model /PATH/TO/LLaMA3/llama3-8b  \
--epochs 20 --log_dir ./log/llama3-8b-w4a4 \
--eval_ppl --wbits 4 --abits 4 --lwc --let --aug_loss --use_matrix --sf 0.1 \
--tasks hendrycksTest,piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande
```

4. evaluation
take LLaMa-7B with W3A16g128 quantization as an example:
```
python main.py \
--method affinequant \
--model /PATH/TO/LLaMA3/  \
--epochs 0 --log_dir ./log/test \
--eval_ppl --wbits 3 --abits 16 --group_size 128 --lwc --let --use_ln_matrix --sf 1e-2 \
--resume /PATH/TO/Pretrained/Parameters 
```

### For LRQuant
```
# W4A4 ppl
python main.py \
--method lrquant \
--model /PATH/TO/LLaMA3/  \
--epochs 20 --log_dir ./log/llama3-8b-w4a4 \
--eval_ppl --wbits 4 --abits 4 --lwc --let \
```

```
# W4A4 zero-shot
python main.py \
--method lrquant \
--model /PATH/TO/LLaMA3/  \
--epochs 20 --log_dir ./log/llama3-8b-w4a4 \
--wbits 4 --abits 4 --lwc --let \
--tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande
```

```
# W4A4 tta
python main.py \
--method lrquant \
--model /PATH/TO/LLaMA3/  \
--epochs 20 --log_dir ./log/llama3-8b-w4a4 \
--eval_ppl --wbits 4 --abits 4 --lwc --let --tta\
```

### For RPTQ
```
python main.py \
--method rptq \
--model /PATH/TO/LLaMA3/  \
--log_dir ./log/llama3-8b-w4a4 \
--eval_ppl --wbits 4 --abits 4 \
--tasks lambada_openai,piqa,arc_easy,arc_challenge,openbookqa,boolq
```

Only quantize KV cache
```
python main.py \
--method rptq \
--model /PATH/TO/LLaMA3/  \
--log_dir ./log/llama3-8b-kv \
--wbits 4 --abits 4 --only_quant_kv \
--eval_ppl --tasks lambada_openai,piqa,arc_easy,arc_challenge,openbookqa,boolq
```


### For Slim++
```
# W2A16G128
python main.py \
--method slim++ \
--model /PATH/TO/LLaMA3/ \
--eval_ppl \
--epochs 50 --log_dir ./log/llama3-8b-w2a16g128 \
--wbits 2 --abits 16 --group_size 128 --lwc \
--tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande

python main.py \
--method slim++ \
--model /PATH/TO/LLaMA3/ \
--eval_ppl \
--epochs 50 --log_dir ./log/llama3-8b-w2a16g128 \
--wbits 2 --abits 16 --group_size 128 --lwc --aug_loss
```

```
# W3A16G128
python main.py \
--method slim++ \
--model /PATH/TO/LLaMA3/ \
--eval_ppl \
--epochs 50 --log_dir ./log/llama3-8b-w3a16g128 \
--wbits 3 --abits 16 --group_size 128 --lwc \
--tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande

python main.py \
--method slim++ \
--model /PATH/TO/LLaMA3/ \
--eval_ppl \
--epochs 50 --log_dir ./log/llama3-8b-w3a16g128 \
--wbits 3 --abits 16 --group_size 128 --lwc --aug_loss
```

More detailed and optional arguments:
- `--model`: the local model path or huggingface format.
- `--wbits`: weight quantization bits.
- `--abits`: activation quantization bits.
- `--group_size`: group size of weight quantization. If no set, use per-channel quantization for weight as default.
- `--lwc`: activate the Learnable Weight Clipping (LWC).
- `--let`: activate the Learnable Equivalent Transformation (LET).
- `--lwc_lr`: learning rate of LWC parameters, 1e-2 as default.
- `--let_lr`: learning rate of LET parameters, 5e-3 as default.
- `--epochs`: training epochs. You can set it as 0 to evaluate pre-trained OmniQuant checkpoints.
- `--nsamples`: number of calibration samples, 128 as default.
- `--eval_ppl`: evaluating the perplexity of quantized models.
- `--tasks`: evaluating zero-shot tasks.
- `--resume`: loading pre-trained OmniQuant parameters.
- `--multigpu`: to inference larger network on multiple GPUs
- `--real_quant`: real quantization, which can see memory reduce. Note that due to the limitations of AutoGPTQ kernels, the real quantization of weight-only quantization can only lead memory reduction, but with slower inference speed.
- `--save_dir`: saving the quantization model for further exploration.



## Related Project
[SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://github.com/mit-han-lab/smoothquant)

[AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://github.com/mit-han-lab/llm-awq)

[GPTQ: Accurate Post-training Compression for Generative Pretrained Transformers](https://github.com/IST-DASLab/gptq)

[AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ)

[OmniQuant:Omnidirectionally Calibrated Quantization for Large Language Models](https://github.com/OpenGVLab/OmniQuant)

[AffineQuant:Affine Transformation Quantization for Large Language Models](https://github.com/bytedance/AffineQuant)

[LRQuant:Learnable and Robust Post-Training Quantization for Large Language Models](https://github.com/zjq0455/RLQ)

[RPTQ: Reorder-Based Post-Training Quantization for Large Language Models](https://github.com/hahnyuan/RPTQ4LLM)
