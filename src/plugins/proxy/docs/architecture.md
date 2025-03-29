# OpenVINO Proxy Plugin architecture

## Motivation

 - OpenVINO may have multiple hardware plugins for similar device type from different vendors (e.g. Intel and NVidia GPUs) and currently user must address them with different names ("GPU" and "NVIDIA" respectively). Using same name for such cases seems to be more user-friendly approach.
 - Moreover, single physical device may have multiple plugin which support it. For example, Intel GPU plugin is OpenCL based, thus can run on other OCL-compatible devices including NVIDIA gpus. In that case we may have primary plugin ("NVIDIA") which provides best performance on target device and fallback plugin ("INTEL_GPU") which helps to improve models coverage for the cases when primary plugin has limited operation set support, and both plugins may be executed via HETERO mode.
 - Implicit HETERO plugin usage may be extended to run on different device types - HETERO:xPU,CPU

 ```mermaid
flowchart TB
    subgraph application [User application]
        direction LR
        GPU.0
        GPU.1
        GPU.2
        GPU.3
    end
    
    subgraph proxy_plugin [Proxy plugin]
        direction LR
        proxy_gpu_0[IntelGPU.0]
        proxy_gpu_1[IntelGPU.1]
        proxy_gpu_2[Hetero:NVIDIA.0,IntelGPU.2]
        proxy_gpu_3[Hetero:NVIDIA.1,IntelGPU.3]
    end
    GPU.0--->proxy_gpu_0
    GPU.1--->proxy_gpu_1
    GPU.2--->proxy_gpu_2
    GPU.3--->proxy_gpu_3
    
    subgraph plugins [Plugins]
        direction LR
        intel_gpu[Intel GPU plugin]
        nvidia[NVIDIA plugin]
        hetero[Hetero plugin]
    end
    proxy_gpu_0--->intel_gpu
    proxy_gpu_1--->intel_gpu
    proxy_gpu_2--->hetero
    proxy_gpu_3--->hetero
    hetero--->intel_gpu
    hetero--->nvidia
    
    subgraph gpu_dev [Intel GPU Plugin devices]
        direction LR
        Intel_GPU.0
        Intel_GPU.1
        Intel_GPU.2
        Intel_GPU.3
    end
    intel_gpu--->Intel_GPU.0
    intel_gpu--->Intel_GPU.1
    intel_gpu--->Intel_GPU.2
    intel_gpu--->Intel_GPU.3
    
    subgraph nvidia_dev [NVIDIA Plugin devices]
        direction LR
        NVIDIA.0
        NVIDIA.1
    end
    
    nvidia--->NVIDIA.0
    nvidia--->NVIDIA.1
    
    
    subgraph hw_dev [Hardware devices]
        direction LR
        dev0[Intel iGPU]
        dev1[Intel dGPU]
        dev2[NVIDIA GPU 1]
        dev3[NVIDIA GPU 2]
    end
    
    Intel_GPU.0 ---> dev0
    Intel_GPU.1 ---> dev1
    Intel_GPU.2 ---> dev2
    Intel_GPU.3 ---> dev3
    
    NVIDIA.0 ---> dev2
    NVIDIA.1 ---> dev3
```

## Plugin responsibilities

 - Providing user-visible aliases which aggregates lower-level plugins ("GPU" alias for "INTEL_GPU" and "NVIDIA_GPU")
 - Hide real hardware plugin names under the common device name
 - Proxy plugin should provide the optimal performance with minimum overhead
 - Implicit HETERO mode run for the cases when multiple plugins can be used for the same device
 - Managing properties to configure target and fallback devices.

## Requirements

 - Do not provide additional libraries and don't affect load time (proxy plugin is a part of openvino library)
 ```mermaid
flowchart LR
    subgraph openvino [openvino library]
        core
        inference
        transformations[Common transformations]
        lp_transformations[LP transformations]
        frontend_common[Frontend common API]
        
        proxy_plugin[Proxy plugin]
        style frontend_common fill:#7f9dc0,stroke:#333,stroke-width:4px
        style transformations fill:#3d85c6,stroke:#333,stroke-width:4px
        style lp_transformations fill:#0b5394,stroke:#333,stroke-width:4px
        style core fill:#679f58,stroke:#333,stroke-width:4px
        style inference fill:#d7a203,stroke:#333,stroke-width:4px
    end
```
 - No overhead for load inference time
    - Fallback to hardware plugin if device is supported only by one plugin
    - Minimal overhead in case of multiple plugins for one device
        - Plus one call of query network if entire model can be executed on preferable plugin
        - Hetero execution in other case
 - Allow to configure device

## Behavior details
 - `ov::Core` can create several instances of proxy plugins (separate instance for each high-level device)
 - Proxy plugin uses `ov::device::uuid` property to match devices from different plugins
 - Plugin allows to set properties of primary plugin, in case of configuration fallback plugin, user should use `ov::device::properties()`
 - Plugin doesn't use the system of devices enumeration of hidden plugins and use ids for enumeration (`DEV.0`, `DEV.1`, ..., `DEV.N` and etc.)
