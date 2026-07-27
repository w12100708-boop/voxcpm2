# voxcpm2-ncnn

VoxCPM2 text-to-speech runtime for ncnn. This repository contains a C++23
runtime library, a small CLI, custom ncnn SDPA support for the DiT graph, and
the Python exporter used to build the runtime asset directory.

## Build

Install xmake, FFmpeg development libraries, and a Vulkan-capable ncnn build.
The xmake project pulls ncnn and uses system FFmpeg for the CLI/server audio
adapter:

```sh
xmake f -m release --profile=false
xmake build voxcpm2
```

Profiling is a compile-time option and is disabled by default. A normal release
build contains no profiler timers or counters and does not expose the CLI
`--profile` flag.

## Assets

Model assets are not committed to this repository. Put exported VoxCPM2 ncnn
assets under `assets/voxcpm2`, or point `-m/--model` at another directory.

Prebuilt fp16 ncnn assets are published at:

<https://huggingface.co/lyrin/voxpm2-ncnn>

The runtime accepts the strict `model.json` schema at `format_version == 2` and
requires the KV-cache decoder components `base_decoder_kv` and
`residual_decoder_kv`. Missing or unknown manifest fields are rejected so the
runtime and exporter schema evolve together. Older
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

`--fp16` is the default and writes compact fp16 ncnn weight files. Components
with backend-specific numerical requirements produce explicit `.cpu` and
`.vulkan` params that share one weight file. Their CPU graphs use fp32 storage;
their Vulkan graphs keep fp16 storage and wrap sensitive RMSNorm operations with
`VoxCPM2DTypeAdapter`.

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

Print component timings by building the optional profiler instrumentation:

```sh
xmake f --profile=true
xmake build voxcpm2
xmake run voxcpm2 -m assets/voxcpm2 -t "你好，欢迎使用 VoxCPM2。" --profile -o out.wav
```

The `SynthesizerConfig::profile` field only has an effect in that instrumented
build. Reconfigure with `xmake f -m release --profile=false` before measuring or
shipping the normal runtime.

Output format is inferred by FFmpeg from `-o/--output`. The CLI does not expose
a max generation length; normal synthesis stops through the exported stop token
head, with an internal runaway cap. `--min-patches` preserves the upstream
`step > min_len` rule: with zero-based generation steps, a value of `N` ignores
stop decisions through step `N`, so the earliest stopped result contains
`N + 2` latent patches.

## Server

Build and start the resident inference server. The model is loaded before Crow
starts listening and remains loaded for the lifetime of the process:

```sh
xmake build voxcpm2-server
xmake run voxcpm2-server --vulkan
```

The default `--threads 0` uses ncnn's CPU topology: physical performance cores
for CPU inference, and a smaller host team plus all logical performance cores
for the CPU-heavy audio VAE portions of Vulkan inference. Pass a positive value
to override the automatic policy for power or thermal constraints.

The default address is `127.0.0.1:8000`. Use `--host 0.0.0.0` to listen on all
interfaces; the server does not provide authentication. Health is available at
`GET /healthz`.

[`POST /v1/audio/speech`](https://developers.openai.com/api/reference/resources/audio/index.md#speech)
follows the core OpenAI Speech API request shape. The
model name is `voxcpm2`; `voice` is required for client compatibility but is a
single-voice placeholder. The default response is MP3, and `mp3`, `opus`,
`aac`, `flac`, `wav`, and `pcm` are supported. PCM is headerless 24 kHz signed
16-bit little-endian audio. Non-empty `instructions`, speeds other than `1.0`,
and SSE streaming are rejected explicitly.

```sh
curl http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"voxcpm2","input":"你好，欢迎使用 VoxCPM2。","voice":"alloy"}' \
  --output speech.mp3
```

OpenAI SDK clients can point their base URL at the local `/v1` endpoint. The
API key is required by the client library but is not checked by this server:

```python
from pathlib import Path
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local")
with client.audio.speech.with_streaming_response.create(
    model="voxcpm2",
    voice="alloy",
    input="你好，欢迎使用 VoxCPM2。",
    response_format="wav",
) as response:
    response.stream_to_file(Path("speech.wav"))
```

Concurrent HTTP connections are accepted, while synthesis requests wait on a
single inference lock so one resident ncnn/Vulkan model is never executed by
multiple requests at once. `--min-patches`, `--timesteps`, and `--cfg-value`
configure synthesis policy for the whole server. In an instrumented
`--profile=true` build, the server also exposes `--profile`; normal builds do
not contain that flag or the profiler instrumentation.

## Vulkan SDPA

The DiT graph uses a local `VoxCPM2SDPA` custom layer so exported graph names do
not overwrite ncnn built-in layers. The layer embeds local GLSL shader sources
with `#embed` and includes flash-attention / cooperative-matrix variants derived
from ncnn SDPA. Vulkan inference uses fp16 storage where supported. Decoder KV
caches remain as native `VkMat` values between steps; each decoder invocation
submits once and downloads only its hidden-state output.

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
- `voxcpm2::read_audio_file`, `voxcpm2::encode_audio`, and
  `voxcpm2::write_audio_file` handle FFmpeg audio I/O.

The `voxcpm2_ncnn` target contains the inference runtime and does not link
FFmpeg. Link `voxcpm2_audio_ffmpeg` in addition to `voxcpm2_ncnn` when using the
audio I/O functions. The `voxcpm2` and `voxcpm2-server` targets already link
both libraries.

The runtime is available in both static and shared forms. The target names are
`voxcpm2_ncnn` and `voxcpm2_ncnn_shared`; their outputs are
`libvoxcpm2_ncnn.a` and `libvoxcpm2_ncnn.so`, respectively.

```sh
xmake build voxcpm2_ncnn_shared
```

## Android JNI

Android arm64 builds expose the synthesis runtime through
`libvoxcpm2_jni.so`, which links against `libvoxcpm2_ncnn.so`. The JNI owner is
`top.yurin.voxcpm2.VoxCPM2`. This repository intentionally provides only the
native binding; the consuming Android project declares the matching methods:

```java
package top.yurin.voxcpm2;

public final class VoxCPM2 {
    static {
        System.loadLibrary("voxcpm2_jni");
    }

    private static native long nativeCreate(
        String modelDir, boolean useVulkan, boolean profile,
        int threads, int vulkanDevice);
    private static native void nativeDestroy(long handle);
    private static native float[] nativeGenerate(
        long handle, String text, String promptText,
        float[] promptAudio, float[] referenceAudio,
        int minPatches, int inferenceTimesteps, float cfgValue);
    private static native void nativeSmokeComponents(long handle);
    private static native int nativeGetInputSampleRate(long handle);
    private static native int nativeGetOutputSampleRate(long handle);
    private static native String[] nativeGetMissingRequiredComponents(long handle);
}
```

`promptAudio` and `referenceAudio` are nullable mono float PCM arrays at
`nativeGetInputSampleRate(handle)`. The returned array is mono float PCM at
`nativeGetOutputSampleRate(handle)`. Calls on one handle are serialized;
`nativeDestroy` waits for an active call and `nativeDestroy(0)` is a no-op.
Invalid arguments raise `IllegalArgumentException`, closed handles raise
`IllegalStateException`, and native runtime failures raise `RuntimeException`.

The Android build requires API 26 or newer and an NDK with Clang 19 or newer
(NDK r28+), because the runtime uses C++23 and `#embed`. Use `c++_shared` when
building the two shared libraries:

```sh
xmake f -p android -a arm64-v8a -m release \
  --ndk=/path/to/android-ndk --ndk_sdkver=26 \
  --runtimes=c++_shared --profile=false
xmake build voxcpm2_ncnn_shared
xmake build voxcpm2_jni
```

An external Android library/app can point xmake-gradle at this repository
without adding a Gradle project here:

```groovy
plugins {
    id 'org.tboox.gradle-xmake-plugin' version '1.2.3'
}

android {
    defaultConfig {
        minSdkVersion 26
        ndk { abiFilters 'arm64-v8a' }
        externalNativeBuild {
            xmake { targets 'voxcpm2_ncnn_shared', 'voxcpm2_jni' }
        }
    }
    externalNativeBuild {
        xmake {
            path '/absolute/path/to/voxcpm-ncnn/xmake.lua'
            sdkver 26
            stl 'c++_shared'
        }
    }
}
```

The model package is not bundled into the native libraries. Download or copy
it to a real app-accessible filesystem directory and pass that directory to
`nativeCreate`; an APK asset path cannot be opened through the runtime's
`std::filesystem` interface. The Android core and JNI targets intentionally do
not include the optional FFmpeg audio adapter.

## License

This project is released primarily under the MIT license in `LICENSE`.
Derivative and embedded third-party scopes are documented in `NOTICE`:

- Apache-2.0 material from reused and rewritten `ncnn_llm` helper code. The
  license text is in `LICENSES/Apache-2.0.txt`.
- BSD-3-Clause material from the local ncnn SDPA Vulkan layer and shader
  derivatives under `src/ncnn_layers`. The license text is in
  `LICENSES/BSD-3-Clause.txt`.
