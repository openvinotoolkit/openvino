# Devstral: the same decoder through an external library

Devstral Small 2505/2507 uses the GGUF `llama` family. Devstral Small 2 and Devstral 2
use `mistral3`. Both families already exist in the native catalog, so the updated frontend
loads their text models without this plugin.

This example demonstrates equivalent external registration. `devstral_decoder` uses the
same `make_decoder_architecture` factory as the native catalog; the entry point explicitly
replaces the existing `llama` and `mistral3` handlers in one frontend instance. It replaces
the whole family handlers, not only files with a Devstral model name. No artificial
`devstral` architecture alias or separate decoder implementation is needed.

Build against an OpenVINO installation containing this branch's shared-builder fixes:

```sh
cmake -S src/frontends/gguf/examples/devstral_extension -B /tmp/devstral-extension \
    -DOpenVINO_DIR=/path/to/openvino/runtime/cmake
cmake --build /tmp/devstral-extension
```

Load the library on the explicitly selected frontend:

```cpp
ov::frontend::FrontEndManager manager;
auto frontend = manager.load_by_framework("gguf");
frontend->add_extension("/tmp/devstral-extension/libgguf_devstral_extension.so");
auto model = frontend->convert(frontend->load("Devstral-Small-2-Q4_K_M.gguf"));
```

Consumers add `GGUFMakeStateful` and `AdaptToGenAI` when cached language-model decoding
is needed. The numerical tests exercise those passes with both native and library routes.

For built-in integration, the factory is already registered for these two family names.
For a genuinely new architecture name, register the same definition in
`builtin_architectures()` and omit the external entry point. The implementation does not
need a rewrite.

The plugin requires the updated shared decoder. It does not retrofit missing YaRN or
attention-temperature behavior into an older runtime. Rebuild developer-SDK extensions
for the OpenVINO version they target. See [support and validation scope](../../docs/devstral_support.md).
