# GGUF extension examples

This standalone project builds three plugins and a CPU runner against a matching
OpenVINO development package containing the GGUF frontend. No OpenVINO rebuild is
needed when building extensions against an already installed compatible package.

| Target | Extension | File contract |
|---|---|---|
| `gguf_projector_extension` | Whole-model `ArchitectureExtension` | `general.architecture = example-projector`; F32 `projection.weight` with GGUF dimensions `[input_width, output_width]` |
| `gguf_decoder_extension` | `ArchitectureExtension` reusing the decoder builder | `general.architecture = example-qwen3`; Qwen3-compatible decoder metadata and tensors |
| `gguf_mmproj_extension` | Two `ProjectorExtension` registrations | `clip` mmproj: `vision / example-gemma3` reuses Gemma3; `audio / example-linear` builds a custom projection branch |

The whole-model projection plugin builds and finishes its own graph. The mmproj
plugin contributes a branch to the coordinator's shared graph. Its custom audio
example takes caller-provided embeddings and applies
`example.audio.projection.weight`; it does not implement waveform preprocessing or
an audio encoder. The Gemma3 alias requires real Gemma3-compatible encoder metadata
and weights. Changing a projector label alone does not make an unrelated model compatible.

## Build

```sh
cmake -S src/frontends/gguf/examples/architecture_extension -B /tmp/gguf-extensions \
    -DOpenVINO_DIR=/path/to/openvino/runtime/cmake
cmake --build /tmp/gguf-extensions --parallel
```

Use the generated library and executable paths for your platform and build configuration.
The commands below use Linux paths. If the dynamic loader cannot find OpenVINO libraries,
source the installation's `setupvars.sh` first. Build against the release that will load
these plugins; the GGUF developer API does not guarantee compatibility across releases.

## Run small synthetic inputs

The fixture generator uses Python's standard library and downloads nothing:

```sh
python3 src/frontends/gguf/examples/architecture_extension/generate_fixtures.py /tmp/gguf-inputs
/tmp/gguf-extensions/gguf_extension_runner \
    /tmp/gguf-extensions/libgguf_projector_extension.so /tmp/gguf-inputs/projection.gguf 3
/tmp/gguf-extensions/gguf_extension_runner \
    /tmp/gguf-extensions/libgguf_mmproj_extension.so /tmp/gguf-inputs/mmproj.gguf 3
/tmp/gguf-extensions/gguf_extension_runner \
    /tmp/gguf-extensions/libgguf_decoder_extension.so /tmp/gguf-inputs/decoder.gguf 3 --stateful
```

Both projection examples return `[1,1,3,3]` embeddings; with the runner's all-one
inputs, each token's values are `[3,7,11]`. The decoder example returns `[1,1,16]` last-token logits and
exercises `GenAIExtension`, which creates stateful caches and adapts IO after frontend normalization.

The runner registers the library before loading the file, converts it, compiles on
CPU, fills supported inputs, checks that F32 outputs are finite, and prints output names, shapes, types and first values.
It supplies synthetic inputs for these fixtures, not tokenization, chat generation,
media preprocessing, or a general multimodal input pipeline. `TOKENS` controls dynamic
token dimensions. Use `--stateful` for the decoder fixture; explicit cache inputs in
stateless decoders require a model-specific harness.

To try the Gemma3 reuse entry, prepare a compatible mmproj GGUF with resolved vision
projector type `example-gemma3` and use `libgguf_mmproj_extension.so`. For real media,
provide the correct preprocessing and positional inputs described in [mmproj.md](../../docs/mmproj.md).

See [extensions.md](../../docs/extensions.md) for whole-decoder builders, operation
conversion extensions, normalization passes, component replacement, and registration timing.
