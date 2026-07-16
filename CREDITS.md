# Credits

## Original ncnn_llm runtime

Some VoxCPM2 runtime glue and tokenizer helper logic reuses and rewrites code
from the Apache-2.0 licensed `ncnn_llm` runtime. That is the derivative-work
scope documented in NOTICE, with the Apache-2.0 text retained in
`LICENSES/Apache-2.0.txt`.

## Native runtime dependencies

- ncnn is distributed under BSD-3-Clause terms. Linux, Android, and Apple
  runtime binaries statically link ncnn; the license text is retained in
  `LICENSES/ncnn-BSD-3-Clause.txt`.
- Linux runtime archives bundle LLVM's OpenMP runtime (`libomp`) under
  Apache-2.0 with LLVM Exceptions. The Apache text and exception are retained
  in `LICENSES/Apache-2.0.txt` and `LICENSES/LLVM-exception.txt`.
- Apple runtime binaries use MoltenVK under Apache-2.0 terms. The Apache text is
  retained in `LICENSES/Apache-2.0.txt`.
