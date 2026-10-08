# LLM Freshman Notes

📚LLM learning notes — before my 7-second memory forgets it all

- base
    + [] transformer
    + [] kv cache
    - GPU Architecture
        + [] 内存层次结构
        + [] 架构发展，从turing到blackwell
    - activation & mlp
        + [] SWiGLU / GeGLU / ReGLU / Situ
        + [] mhc
- online softmax
- attention
    + [] self attention
    + [] online attention
    + [] flash attention
    + [] flash attention 2
    + [] flash attention 3
    + [] flash decoding
    + [] flash decoding++
    + [] multi-head attention (MHA)
    + [] multi-query attention (MQA)
    + [] group-query attention (GQA)
    + [] multi-head latent attention (MLA)
    + [] page attention
    + [] ring attention
    + [] linear attention
    + [] native sparse attention
- kv cache opt
    + sparse (DSA)
    + quantization
    + share
    + windows (SWA)
- norm
    + Batch Norm
    + Layer Norm 
    + RMS Norm
- position embedding
    + [] Rope
    + [] AliBi
- quantization
    + [] smooth quant
    + [] AWQ
    + [] KIVI
    + [] GPTQ
    + [] FP8 (E4M3、E5M2)
    + [] FP4 / FP6 / NF4
    + [] NVFP4 / MXFP4 / MXFP8
- speculative decoding
    + [] Eagle 1 2 3
    + [] Multi-token prediction (MTP)
    + [] dspark
    + [] dflash
- Inference design
    + [] chunked prefill
    + [] continous batching
    + [] sliding windows
    + [] CUDA Graph
    + [] vllm
    + [] sglang
- Reinforcement learning
    + [] PPO
    + [] GRPO
    + [] DAPO
    + [] DPO
    + [] SFT
- distributed parallel
    + [] data Parallelism (DP, Zero 1、2、3)
    + [] Tensor Parallelism (TP)
    + [] Pipeline Parallelism (PP, 1F1B, Zero-Bubble)
    + [] Sequence Parallelism (SP)
    + [] Context Parallelism (CP)
    + [] Hardware Topology (NVLink, NVSwitch, PCIe, InfiniBand, RoCE)
- Mixture of experts (MOE)
    + [] native
    + [] deepgemm megaMoe
    + [] MoonEp
- open source
    + [] deepgemm
    + [] flash attention
    + [] flashinfer
    + [] marlin
    + [] b12x

- gemm
    + [] gemm opt
    + [] split-k / stream-k

- dsl
    + [] cuda
    + [] PTX
    + [] triton
    + [] cutedsl
    + [] tilelang
    + [] cutile
