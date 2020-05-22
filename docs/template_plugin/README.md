# template-plugin

Template Plugin for Inference Engine which demonstrates basics of how Inference Engine plugin can be built and implemented on top of Inference Engine Developer Package and Plugin API.

## How to build

```bash
$ cd $DLDT_HOME
$ mkdir $DLDT_HOME/build
$ cd $DLDT_HOME/build
$ cmake -DENABLE_TESTS=ON -DENABLE_BEH_TESTS=ON -DENABLE_FUNCTIONAL_TESTS=ON ..
$ make -j8
$ cd $TEMPLATE_PLUGIN_HOME
$ mkdir $TEMPLATE_PLUGIN_HOME/build
$ cd $TEMPLATE_PLUGIN_HOME/build
$ cmake -DInferenceEngineDeveloperPackage_DIR=$DLDT_HOME/build ..
$ make -j8
```
