
Include:
| Methods | Quantize | PPL Eval | Task Eval | Save |
| :--- | ---: | :---: | :---: | :---: 
| OmniQuant | ✅ | ✅ | TODO | ✅ 
| AffineQuant | ✅ | ✅ | TODO | TODO 
| LRQuant | TODO | TODO | TODO | TODO 

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
--eval_ppl --wbits 3 --abits 16 --lwc --let --use_ln_matrix --sf 1e-2

# W3A16g128
python main.py \
--method affinequant \
--model /PATH/TO/LLaMA3/llama3-8b  \
--epochs 20 --log_dir ./log/llama3-8b-w3a16g128 \
--eval_ppl --wbits 3 --abits 16 --group_size 128 --lwc --let --use_ln_matrix --sf 1e-2
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
--model /PATH/TO/LLaMA3/llama3-8b  \
--epochs 20 --output_dir ./log/llama3-8b-w4a4 \
--eval_ppl --wbits 4 --abits 4 --lwc --let \
```

```
# W4A4 zero-shot
python main.py \
--model /PATH/TO/LLaMA3/llama3-8b  \
--epochs 20 --output_dir ./log/llama3-8b-w4a4 \
--wbits 4 --abits 4 --lwc --let \
--tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande
```

```
# W4A4 tta
python main.py \
--model /PATH/TO/LLaMA3/llama3-8b  \
--epochs 20 --output_dir ./log/llama3-8b-w4a4 \
--eval_ppl --wbits 4 --abits 4 --lwc --let --tta\
```

### For RPTQ
```
python main.py \
--model /PATH/TO/LLaMA3/llama3-8b  \
--output_dir ./log/llama3-8b-w4a4 \
--eval_ppl --wbits 4 --abits 4 \
--tasks lambada_openai,piqa,arc_easy,arc_challenge,openbookqa,boolq
```

Only quantize KV cache
```
python main.py \
--model /PATH/TO/LLaMA3/llama3-8b  \
--wbits 4 --abits 4 --only_quant_kv \
--eval_ppl --tasks lambada_openai,piqa,arc_easy,arc_challenge,openbookqa,boolq
```



## Related Project
[SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://github.com/mit-han-lab/smoothquant)

[AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://github.com/mit-han-lab/llm-awq)

[GPTQ: Accurate Post-training Compression for Generative Pretrained Transformers](https://github.com/IST-DASLab/gptq)

[AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ)

[OmniQuant:Omnidirectionally Calibrated Quantization for Large Language Models](https://github.com/OpenGVLab/OmniQuant)

[AffineQuant:Affine Transformation Quantization for Large Language Models](https://github.com/bytedance/AffineQuant)

[LRQuant:Learnable and Robust Post-Training Quantization for Large Language Models](https://github.com/zjq0455/RLQ)

[RPTQ: Reorder-Based Post-Training Quantization for Large Language Models](https://github.com/hahnyuan/RPTQ4LLM)
