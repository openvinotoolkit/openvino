# OpenVINO™ Developer Documentation

Welcome to the OpenVINO™ Toolkit Developer Documentation. This documentation helps to deeper understand the ideology of OpenVINO, provides the complete information about ideas which are applied inside.

The OpenVINO™ Toolkit provides a functionality to load model from different frameworks and run it on different accelerators.

```mermaid
flowchart LR
    subgraph models [Models]
        ir[("IR (*.xml)")]
        onnx[("ONNX (*.onnx)")]
        paddle[("PaddlePaddle (*.pdmodel)")]
        tf[("Tensorflow (*.pb)")]
        
        style ir fill:#427cb0
        style onnx fill:#427cb0
        style paddle fill:#427cb0
        style tf fill:#427cb0
        
        style models fill:#aeb6ba
        
        click onnx "https://github.com/onnx/onnx"
    end
    
    mo{{Model Optimizer}}
    
    style mo fill:#6c9f7f
    
    onnx--convert--->mo
    paddle--convert--->mo
    tf--convert--->mo
    mo--->ir
    
    subgraph plugins [OV Plugins]
        auto(["AUTO"])
        cpu(["Intel_CPU"])
        gpu(["Intel_GPU"])
    end
    subgraph frontends [OV Frontends]
        ir_fe["IR Frontend"]
        onnx_fe["ONNX Frontend"]
        paddle_fe["Paddle Frontend"]
    end
    openvino(openvino library)
    ir--Read ir---ir_fe
    onnx--Read onnx--- onnx_fe
    paddle--Read paddle---paddle_fe
    ir_fe--->openvino
    onnx_fe--->openvino
    paddle_fe--->openvino
    
    openvino--infer--->cpu
    openvino--infer--->gpu
    openvino--infer--->auto
```

## [Get started](./get_started.md)

 * [Build OpenVINO](./build.md)
 * How to:
     * [Add new operation](../../src/core/docs/operation_enabling_flow.md)
     * [Add new transformation](#todo)
     * [Get code coverage report](./test_coverage.md) 
     * [Add component developer documentation](./dev_doc_guide.md)
 * [OpenVINO contribution guideline](../../CONTRIBUTING.md)
 * [OpenVINO Debug Capabilities](./debug_capabilities.md)

## OpenVINO repository structure

The repository is organized in such a way that the components contain all dependencies inside (for example, thirdparty, tests, documentation, and other). 
With this approach, groups of developers, who are concentrated on different components, can efficiently work together.

The OpenVINO Repository compositions from next components, click on the component name to get more information about it:
<pre>
 <code>
 <a href="../../README.md">openvino/</a>                  // OpenVINO Repository
    .ci/                    // CI settings for Azure
    .github/                // Github actions and PR templates
    cmake/                  // Global CMake scripts
    docs/                   // OpenVINO documentation
    licensing/              // Licenses
    samples/                // OpenVINO samples
    scripts/                // Helper scripts
    <a href="../../src/README.md">src/</a>                    // Folder with core OpenVINO components
    tests/                  // Infrastructure tests, which validate full pipelines
    thirdparty/             // Common third-party dependencies
    tools/                  // OpenVINO tools
 </code>
</pre>

### OpenVINO Component Structure

The OpenVINO component should contain all dependencies inside (for example, thirdparty, tests, documentation, and other). The abstract component structure with comments and marks for optional folders is presented below.

```
ov_component/           // Component folder
    cmake/              // (optional) CMake scripts which are related only for this component
    dev_api/            // (optional) Developer API is used in case if component wants to provide some API for internal developers
    docs/               // (optional) Contains detailed component documentation
    include/            // (optional) Public component API
    src/                // Sources of the component
    tests/              // Tests on the component
    thirdparty/         // (optional) Third-party dependencies
    CMakeLists.txt      // Main CMake script
    README.md           // (optional) Endtry point for the developer documentation
```


## Features

 * [Conditional Compilation](./conditional_compilation.md)

## See also
 * [OpenVINO™ README](../../README.md)
