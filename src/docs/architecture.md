# OpenVINO Architecture

This guide encompasses existed architectural ideas and guidelines for architecture development.

## Main architectual design concepts

OpenVINO development is based on next principals:
 * Performance and Scalability - OpenVINO should provide the optimal performance in different customer cases
 * Availability and Resilience - OpenVINO is high quality product and we should demonstrate this high level in different cases
 * Security - OpenVINO is applied in different applications, it is why we have a strict validation pipelines which allow to find vulnerabilities on the development stage.

Each OpenVINO component is projected with using DOTADIW (Do One Thing And Do It Well) approach, it means that OpenVINO is highly granularity product and we are avoiding mixing of responsibilities for different components.

### OpenVINO Component responsibilities

```mermaid
flowchart TB
    subgraph tools [Tools]
        mo{{Model Optimizer}}
        pot{{PoT}}
    
        style mo fill:#6c9f7f
        style pot fill:#6c9f7f
    end
    subgraph tutorials [Tutorials]
        samples[Samples]
        demos[Demos]
        notebooks[Notebooks]
    end
    subgraph api [API]
        cpp[C++]
        subgraph bindings [Bindings]
            c_api[C]
            python_api[Python]
        end
    end
    
    subgraph plugins [OV Plugins]
        auto(["AUTO"])
        cpu(["Intel_CPU"])
        gpu(["Intel_GPU"])
    end
    subgraph frontends [OV Frontends]
        direction TB
        ir_fe["IR Frontend"]
        onnx_fe["ONNX Frontend"]
        paddle_fe["Paddle Frontend"]
    end
    openvino(openvino library)
    
    frontends--Read model--->openvino
    openvino--API--->api
    openvino--infer--->plugins
```

All OpenVINO components can be logicaly divided to several groups:

 * **OpenVINO library** is a core library which provides the OpenVINO API.
 * **Bindings** are libraries which provide OpenVINO API for different languages.
 * **Frontends** are libraries which converts model from framework representation to OpenVINO format.
 * **Plugins** are components which allow to run models on different accelerators.
 * **Tools** a set of different components which provide additional opportunities for OpenVINO.
 * **Tutorials** different applications which show how user can work with OpenVINO.

### OpenVINO Inference pipeline

OpenVINO inference pipeline has several objects which are responsible for different stages of model inference.

 * `Core` object is a global context for Model inference, allows to create the unique environment for inference pipeline.
     * `read_model()` finds the right frontend and converts Framework model to OpenVINO representation.
     * `compile_model()` loads hardware plugin and compiles model on the device.
 * `CompiledModel` represents the hardware specified model, this model can contain additional plugin optimization and on this stage model already was compiled for the specific device.
 * `InferRequest` provides an interface for running inference.

```mermaid
flowchart TB
    subgraph core [ov::Core core]
        read_model["core.read_model()"]
        compile_model["core.compile_model()"]
    end
    fw_model[(Framework model)]
    ov_model[ov::Model]
    
    subgraph compiled_models [Compiled models]
        cpu_model["CPU"]
        gpu_model["GPU"]
        auto_model["AUTP"]
    end
    subgraph infer_reguests [InferRequests]
        cpu_req["CPU"]
        gpu_req["GPU"]
        auto_req["AUTP"]
    end
   
    result[Results]
    
    fw_model--->read_model
    read_model--->ov_model
    ov_model-->compile_model
    
    fw_model--->compile_model
    
    compile_model--"compile_model(CPU)"--->cpu_model
    compile_model--"compile_model(GPU)"--->gpu_model
    compile_model--"compile_model(AUTO)"--->auto_model
    
    cpu_model--"create_infer_request()"--->cpu_req
    gpu_model--"create_infer_request()"--->gpu_req
    auto_model--"create_infer_request()"--->auto_req
    
    cpu_req--"infer()"-->result
    gpu_req--"infer()"-->result
    auto_req--"infer()"-->result
```

## See also
 * [OpenVINO Developer documentation](../../docs/dev/index.md)
