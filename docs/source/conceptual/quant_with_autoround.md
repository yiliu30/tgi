### TGI Intro

- Supported quantization algos: awq, gptq, bitsandbytes, FP8...

### Usage Intro
There are two ways to use quantization:
1. Load the already quantized model
```bash
 # select the quantization method by `--quantize`
docker run \
    --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:latest\
    --model-id $model \
    --quantize gptq # <------------
```

2. Quantize a float model and save it to local
```bash
text-generation-server quantize path/to/float/model/ /path/to/save/quantized/model
# Add --upload-to-model-id username/model_id to push the created model to the hub directly
```
-  Only support quantizing float model with `GPTQ`.

### Implementation Details

Implement a `QLinear` module in-tree, call the kernel out-of-tree.

The `get_linear` is the entry to replace the `Linear` with the appropriate `QLinear` based on the selected algorithm name.

#### GPTQ
- `gptq.exllama.ExllamaQuantLinear`, using [exllama.q4_matmul](https://github.com/turboderp/exllama) as kernel
- `gptq.quant_linear.QuantLinear`, using triton kernel `matmul_248_kernel`

#### AWQ
- `awq.WQLinear`, using [awq_inference_engine.gemm_forward_cuda](https://github.com/mit-han-lab/llm-awq) as kernel
- Convert the weight into `GPTQ`-like format by `fast_awq_to_gptq`?

<!-- #### Others
- marlin, `MarlinLinear`, call [marlin.mul](https://github.com/IST-DASLab/marlin) -->


### Q&A
Q: Does the new quantization algo support depend on the transformers?

A: No, it has own logic to load or quantize the quantized model.


------------------------------


## Proposal

### Goals
- Support inference with quantized models using Auto-round.
- Support quantizing the float model with Auto-round (Nice to have).

### Support inference with quantized models using Auto-round
- Users can use the already quantized model by specifying `--quantize` with `autoround`.
- We need to quantize the model with auto-round and export it with `GPTQ` format, and upstream it to Hugging Face hub.

- Usage
    ```bash
    text-generation-launcher \
    --model-id INC/Llama-2-7b-Chat-Autoround \
    --trust-remote-code --port 8080 \
    --max-input-length 3072 --max-total-tokens 4096 --max-batch-prefill-tokens 4096 \
    --quantize auto-round   # <---------------- 

    ```

### Support quantizing the float model with Auto-round (Nice to have)
- Users can use TGI to quantize the model with auto-round and save it to locally.
    ```bash
    text-generation-server quantize \
        --MODEL_ID path/to/float/model/\
        --OUTPUT_DIR /path/to/save/quantized/model \
        --method autoround
    ```

### Support model scope and quantization configurations
- Align with auto-round support list
- Model list: ...
- configurations list: ...

-- End of Proposal
------------------------------
#### TGI Usage demo

```bash
docker run  -p 8080:80 -v $volume:/data --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none -e http_proxy=http://child-ir.intel.com:912 -e https_proxy=http://child-ir.intel.com:912 --cap-add=sys_nice --ipc=host ghcr.io/huggingface/tgi-gaudi:2.0.0 --model-id $model --max-input-tokens 1024 --max-total-tokens 2048
```

```bash
# usage 
text-generation-launcher \
--model-id TheBloke/Llama-2-7b-Chat-AWQ \
--trust-remote-code --port 8080 \
--max-input-length 3072 --max-total-tokens 4096 --max-batch-prefill-tokens 4096 \
--quantize awq # <------------
```

------------------------------




### Some useful links
- Add marlin quant, https://github.com/huggingface/text-generation-inference/pull/2014
- Add AWQ
    - https://github.com/huggingface/text-generation-inference/issues/781
    - https://github.com/huggingface/text-generation-inference/pull/1054
