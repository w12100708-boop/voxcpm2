# Credits

## vLLM

The Paged KV Cache design in this repository is inspired by vLLM's
PagedAttention and KV cache block-table architecture. The implementation here
is a clean C++23/ncnn implementation and does not copy vLLM source code. The
vLLM project is credited as a design reference only; its Apache-2.0 license
does not define the license scope of this repository's Paged KV Cache code.

- Project: https://github.com/vllm-project/vllm
- PagedAttention design: https://docs.vllm.ai/en/latest/design/paged_attention/

## Original ncnn_llm runtime

Some VoxCPM2 runtime glue and tokenizer helper logic reuses and rewrites code
from the Apache-2.0 licensed `ncnn_llm` runtime. That is the derivative-work
scope documented in NOTICE, with the Apache-2.0 text retained in
`LICENSES/Apache-2.0.txt`.
