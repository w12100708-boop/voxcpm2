# voxcpm2-ncnn

VoxCPM2 text-to-speech runtime for ncnn. The project contains a C++23 library, a small CLI, and the Python exporter needed to produce the ncnn asset directory.

## Build

Install ncnn dependencies, FFmpeg development libraries, and xmake. The C++ build uses system FFmpeg through xmake:

```sh
xmake f -m release
xmake build voxcpm2
```

## Assets

Model assets are not committed to this repository. Put the exported VoxCPM2 ncnn assets under `assets/voxcpm2`, or point `-m/--model` at another directory.

Prebuilt ncnn assets are published at <https://huggingface.co/lyrin/voxpm2-ncnn>.
The current runtime requires `model.json` `format_version >= 2`, exported with `base_decoder_kv` and `residual_decoder_kv`; older `base_decoder_step` / `residual_decoder_step` assets must be re-exported.

```sh
huggingface-cli download lyrin/voxpm2-ncnn --local-dir assets
```

Prepare metadata from a Hugging Face snapshot:

```sh
uv run tools/export_assets.py --snapshot /path/to/openbmb/VoxCPM2/snapshot --out assets/voxcpm2
```

Export the ncnn components:

```sh
uv run tools/export_components.py --asset-dir assets/voxcpm2 --no-fp16
```

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

## Tests

```sh
xmake test
```

## Library

The public C++ API lives under `include/voxcpm2`:

- `voxcpm2::Tokenizer` exposes `encode(text)`.
- `voxcpm2::Synthesizer` loads the ncnn asset directory and returns `AudioBuffer`.
- `voxcpm2::read_audio_file` and `voxcpm2::write_audio_file` handle FFmpeg audio I/O.

## License

This project is released primarily under the MIT license in `LICENSE`. A small Apache-2.0 derivative scope is retained only for reused and rewritten `ncnn_llm` helper code described in `NOTICE`; the Apache-2.0 text is in `LICENSES/Apache-2.0.txt`.
