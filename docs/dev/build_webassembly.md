# Build OpenVINO™ Runtime for WebAssembly

This guide shows how to build OpenVINO for WebAssembly using  [Emscripten SDK](https://emscripten.org/). Emscripten SDK can be directly downloaded and used, but it is more easier to use the [emscripten/emsdk](https://hub.docker.com/r/emscripten/emsdk) docker image. 

The approach is validated on Linux, Windows and macOS operation systems.

## Software Requirements

- [Docker Engine](https://docs.docker.com/engine/install/)

## How to build

1. Clone OpenVINO repository and init submodules:
```sh
git clone https://github.com/openvinotoolkit/openvino.git
cd openvino
git submodule update --init --recursive
```
2. Run docker image and mount a volume with OpenVINO source code:
```sh
$ docker pull emscripten/emsdk
$ docker run -it --rm -v `pwd`:/openvino emscripten/emsdk bash
```
3. (CMake configure) Run cmake configure step using helper emscripten command:
```sh
$ mkdir build && cd build
$ emcmake cmake -DCMAKE_BUILD_TYPE=Release /openvino
```
4. (CMake build) Build OpenVINO project:
```sh
$ emmake make -j$(nproc)
```
`openvino.wasm` and `openvino.js` files are located in:
- `<openvino_source_dir>/bin/ia32/Release/` on host machine file system.
- `/openvino/bin/ia32/Release` in docker environment.
These files can be used in browser applications. 

## See also

 * [OpenVINO README](../../README.md)
 * [OpenVINO Developer Documentation](index.md)
 * [OpenVINO Get Started](./get_started.md)
 * [How to build OpenVINO](build.md)

