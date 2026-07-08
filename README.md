# voxcpm2-ncnn

VoxCPM2 text-to-speech runtime for ncnn. This repository contains a C++23
runtime library, a small CLI, custom ncnn SDPA support for the DiT graph, and
the Python exporter used to build the runtime asset directory.

## Build

Install xmake, FFmpeg development libraries, and a Vulkan-capable ncnn build.
The xmake project pulls ncnn and uses system FFmpeg:

```sh
xmake f -m release
xmake build voxcpm2
```

## Assets

Model assets are not committed to this repository. Put exported VoxCPM2 ncnn
assets under `assets/voxcpm2`, or point `-m/--model` at another directory.

Prebuilt fp16 ncnn assets are published at:

<https://huggingface.co/lyrin/voxpm2-ncnn>

The runtime requires `model.json` `format_version >= 2` and the KV-cache
decoder components `base_decoder_kv` and `residual_decoder_kv`. Older
`base_decoder_step` / `residual_decoder_step` assets are not used by this
runtime.

```sh
huggingface-cli download lyrin/voxpm2-ncnn --local-dir assets/voxcpm2
```

Prepare metadata from a Hugging Face snapshot:

```sh
uv run tools/export_assets.py --snapshot /path/to/openbmb/VoxCPM2/snapshot --out assets/voxcpm2
```

Export the ncnn components:

```sh
uv run tools/export_components.py --asset-dir assets/voxcpm2 --fp16
```

`--fp16` is the default and writes compact fp16 ncnn weight files. At runtime,
the Vulkan path still keeps numerically sensitive decoder, feature encoder, and
DiT estimator components on fp32 storage/packing options. This preserves audio
quality while still using the fp16 asset package.

## CLI

Zero-shot synthesis:

```sh
xmake run voxcpm2 -m assets/voxcpm2 -t "你好，欢迎使用 VoxCPM2。" -o out.wav
```

Prompt continuation:

```sh
xmake run voxcpm2 -m assets/voxcpm2 -t "这是续写测试" --prompt "你好" --prompt-audio prompt.wav -o out.wav
```

Reference audio:

```sh
xmake run voxcpm2 -m assets/voxcpm2 -t "这是参考音频测试" --reference-audio voice.wav -o out.flac
```

Enable Vulkan:

```sh
xmake run voxcpm2 -m assets/voxcpm2 -t "你好，欢迎使用 VoxCPM2。" --vulkan -o out.wav
```

Smoke-test exported components:

```sh
xmake run voxcpm2 -m assets/voxcpm2 --smoke-components
xmake run voxcpm2 -m assets/voxcpm2 --smoke-components --vulkan
```

Print coarse runtime timings by enabling the optional profile build:

```sh
xmake f --profile=true
xmake build voxcpm2
xmake run voxcpm2 -m assets/voxcpm2 -t "你好，欢迎使用 VoxCPM2。" --profile -o out.wav
```

Output format is inferred by FFmpeg from `-o/--output`. The CLI does not expose a max generation length; normal synthesis stops through the exported stop token head, with an internal runaway cap.

## Vulkan SDPA

The DiT graph uses a local `VoxCPM2SDPA` custom layer so exported graph names do
not overwrite ncnn built-in layers. The layer embeds local GLSL shader sources
with `#embed` and includes flash-attention / cooperative-matrix variants derived
from ncnn SDPA. The current production options keep the sensitive SDPA-bearing
components off fp16 storage/packing, so those fast-attention and cooperative
matrix kernels are present but normally not selected on the known-good audio
path.

For preset-based tensor comparisons, use the Python helper. By default it
compares the safe Vulkan preset against the full-fp16 Vulkan preset; pass
`--base-preset`, `--target-preset`, and `--scan` for broader blob scans.

```sh
uv run tools/compare_ncnn.py --asset-dir assets/voxcpm2 --component dit_estimator
```

## Tests

```sh
xmake test -j1
```

## Library

The public C++ API lives under `include/voxcpm2`:

- `voxcpm2::Tokenizer` exposes `encode(text)`.
- `voxcpm2::Synthesizer` loads the ncnn asset directory and returns `AudioBuffer`.
- `voxcpm2::read_audio_file` and `voxcpm2::write_audio_file` handle FFmpeg audio I/O.

## License

This project is released primarily under the MIT license in `LICENSE`.
Derivative and embedded third-party scopes are documented in `NOTICE`:

- Apache-2.0 material from reused and rewritten `ncnn_llm` helper code. The
  license text is in `LICENSES/Apache-2.0.txt`.
- BSD-3-Clause material from the local ncnn SDPA Vulkan layer and shader
  derivatives under `src/ncnn_layers`. The license text is in
  `LICENSES/BSD-3-Clause.txt`.
