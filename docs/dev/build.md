# How to build OpenVINO

```mermaid
gantt 
    %% Use a hack for centry as a persantage
    dateFormat YYYY
    axisFormat %y
    todayMarker off
    title       OpenVINO getting started pipeline
    Setup environment :env, 2000, 1716w
    Build openvino :crit, build, after env, 1716w
    Run tests :active, run, after build, 1716w
```

The articles below provide the basic informations about the process of building OpenVINO.

* [Windows](build_windows.md)
* [Linux](build_linux.md)
* [Mac (Intel CPU)](build_mac_intel_cpu.md)
* [Mac (ARM)](build_mac_arm.md)
* [Android](build_android.md)
* [Raspbian Stretch](./build_raspbian.md)
* [Web Assembly](./build_webassembly.md)
* [Docker Image](https://github.com/openvinotoolkit/docker_ci/tree/master/dockerfiles/ubuntu18/build_custom)
* [Linux RISC-V](./build_riscv64.md)

> **NOTE**: For the details on how to build static OpenVINO, refer to [Building static OpenVINO libraries](static_libaries.md)

## See also

 * [OpenVINO README](../../README.md)
 * [OpenVINO Developer Documentation](index.md)
 * [OpenVINO Get Started](./get_started.md)

