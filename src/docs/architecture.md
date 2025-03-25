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
        ovc{{OpenVINO model converter}}
    
        style ovc fill:#6c9f7f
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
        auto_model["AUTO"]
    end
    subgraph infer_reguests [InferRequests]
        cpu_req["CPU"]
        gpu_req["GPU"]
        auto_req["AUTO"]
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

### OpenVINO Components Relationships

The diagram below shows internal components and their relationships inside the OpenVINO dynamic library.

```mermaid
flowchart LR
    subgraph legend [Legend]
    interface_lib[Interface library]
    style interface_lib fill:#af8401

    object_lib[Object library]
    style object_lib fill:#6e9f01

    static_lib[(Static library)]
    style static_lib fill:#01429f
    dynamic_lib([Dynamic library])
    style dynamic_lib fill:#cb6969

    some_api{{Some library API}}
    some_dev_api{{Some developer API}}

    object_lib ===>|Link with| interface_lib

    some_api --->|Depends on headers only| static_lib
    some_api -.->|Include headers| some_dev_api
    end
```
```mermaid
flowchart TB
    util[(openvino::util)]
    style util fill:#01429f

    itt[(openvino::itt)]
    style itt fill:#01429f

    pugixml[(openvino::pugixml)]
    style pugixml fill:#01429f

    builders[(builders)]
    style builders fill:#01429f

    reference[(reference implementations)]
    style reference fill:#01429f

    shape_inference[(shape inference)]
    style shape_inference fill:#01429f

    subgraph core [core]
        style core fill:#6e9f01
        core_api{{Core public API}}

        core_api--->builders
        core_api--->reference
        core_api--->shape_inference
    end

    subgraph core_dev [openvino::core::dev]
        style core_dev fill:#af8401
        core_dev_api{{Core developer API}}
    end

    core ===> core_dev

    core_dev_api-.->openvino_dev_api

    pugixml ===> core
    builders ===> core
    reference ===> core
    shape_inference ===> core
    util ===> core
    itt ===> core

    subgraph inference [inference]
        style inference fill:#6e9f01
        inference_api{{Inference public API}}
        inference_dev_api{{Infernce developer API}}

        inference_api-.->inference_dev_api
    end

    core ===> inference
    util ===> inference
    itt ===> inference

    inference_dev_api-.->openvino_dev_api

    subgraph frontend_common [frontend_common]
        style frontend_common fill:#6e9f01
        frontend_api{{Frontend API}}
    end

    core_dev ===> frontend_common
    util ===> frontend_common


    frontend_common ---> core_dev


    subgraph transformations [OpenVINO transformations]
        style transformations fill:#6e9f01
        transformations_api{{Transformations API}}
    end

    reference ===> transformations
    builders ===> transformations
    shape_inference ===> transformations
    itt ===> transformations
    core_dev ===> transformations

    transformations ---> core_dev

    subgraph lp_transformations [OpenVINO LP transformations]
        style lp_transformations fill:#6e9f01
        lp_transformations_api{{LP transformations API}}
    end

    itt ===> lp_transformations
    transformations ---> lp_transformations
    core ---> lp_transformations


    subgraph openvino [openvino::runtime]
        style openvino fill:#cb6969
        openvino_api{{OpenVINO Public API}}
    end

    transformations ===> openvino
    lp_transformations ===> openvino
    frontend_common ===> openvino
    reference ===> openvino
    builders ===> openvino
    shape_inference ===> openvino
    core ===> openvino

    frontend_api-.->openvino_api
    inference_api-.->openvino_api
    core_api-.->openvino_api

    subgraph openvino_dev [openvino::runtime::dev]
        style openvino_dev fill:#af8401
        openvino_dev_api{{OpenVINO developer API}}
    end

    openvino ===> openvino_dev
    inference_dev_api-.->openvino_dev_api
    lp_transformations_api-.->openvino_dev_api
```

## See also
 * [OpenVINO Developer documentation](../../docs/dev/index.md)
