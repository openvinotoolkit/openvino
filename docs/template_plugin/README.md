# OpenVINO Template Plugin

Template Plugin for OpenVINO™ Runtime which demonstrates basics of how OpenVINO™ Runtime plugin can be built and implemented on top of OpenVINO Developer Package and Plugin API.
As a backend for actual computations OpenVINO reference implementations is used, so the Template plugin is fully functional.

## How to build

```bash
$ cd $OPENVINO_HOME
$ mkdir $OPENVINO_HOME/build
$ cd $OPENVINO_HOME/build
$ cmake -DENABLE_TESTS=ON -DENABLE_FUNCTIONAL_TESTS=ON ..
$ make -j8
$ cd $TEMPLATE_PLUGIN_HOME
$ mkdir $TEMPLATE_PLUGIN_HOME/build
$ cd $TEMPLATE_PLUGIN_HOME/build
$ cmake -DOpenVINODeveloperPackage_DIR=$OPENVINO_HOME/build ..
$ make -j8
```
