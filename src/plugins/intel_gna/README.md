# GNA Plugin Developer Documentation

Welcome to the GNA Plugin Developer Documentation. This documentation helps deeper understand the GNA Plugin architecture and gives detailed information on the concepts and ideas used inside.

The GNA plugin provides a way to run inference on Intel® GNA, as well as in the software execution mode on CPU.

## Get Started
 * [Build](./build.md)
 * How to:
    * [GNA configuration](https://docs.openvino.ai/latest/openvino_docs_install_guides_configurations_for_intel_gna.html)
    * [Add new transformation](#todo)
 * [debug capabilities](./docs/debug_capabilities.md)
 * [Code Style](./docs/code_style.md)

## Key contacts
* Denis Orlov <denis.orlov@intel.com>
* Mikhail Ryzhov <mikhail.ryzhov@intel.com>
* Evgeny Kotov <evgeny.kotov@intel.com>
* Szymon Jakub Irzabek <szymon.jakub.irzabek@intel.com>
* Nadezhda Ageeva <nadezhda.ageeva@intel.com>

## Architecture

### Plugin High Level Design
<img src="./docs/hld.png" title="High level plugin architecture:">

Detailed component description:
* Configuration
* Logger
* Transformations
* Model Quantizer
* Custom operations
* FP32 operations
* Operation validators
* Inputs info
* Outputs info
* Pre/Post processing
* Graph Compiler
* Serializer
* Memory Allocator
* Request Factory
* Device Helper

### Network loading flow
<img src="./docs/load_network_flow.png">

### Plugin Component Structure

The OpenVINO component contains all dependencies (for example, third-party, tests, documentation, and others). An example component structure with comments and marks for optional folders is presented below.

```
intel_gna/              // Plugin folder
    cmake/              // CMake scripts that are related only to the GNA plugin
    docs/               // Contains detailed plugin documentation
    src/                // Sources of the plugin
    tests/              // Tests for the plugin
        functional/     // Functional plugin tests
        unit/           // Unit plugin tests
    CMakeLists.txt      // Main CMake script
    README.md           // Entry point for the developer documentation
```


## Features
 * [Import/Export](#todo)

## See Also

 * [OpenVINO README](../../README.md)