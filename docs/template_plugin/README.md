# OpenVINO Template Plugin

Template Plugin for OpenVINO™ Runtime which demonstrates basics of how OpenVINO™ Runtime plugin can be built and implemented on top of OpenVINO Developer Package and Plugin API.
As a backend for actual computations OpenVINO reference implementations is used, so the Template plugin is fully functional.

## How to build

```bash
$ cd <openvino_dir>
$ mkdir <openvino_dir>/build
$ cd <openvino_dir>/build
$ cmake -DENABLE_TESTS=ON -DENABLE_FUNCTIONAL_TESTS=ON ..
$ make -j8
$ cd <template_plugin_dir>
$ mkdir <template_plugin_dir>/build
$ cd <template_plugin_dir>/build
$ cmake -DOpenVINODeveloperPackage_DIR=<openvino_dir>/build -DENABLE_TEMPLATE_REGISTRATION=ON ..
$ make -j8
```

`ENABLE_TEMPLATE_REGISTRATION` cmake options registers the plugin in plugin.xml configuration file and enables install target for the plugin.
