# How to enable conditional compilation for new component

## Contents

- [Introduction of conditional compilation macro](#Introduction-of-conditional-compilation-macro)
- [Enable conditional complication for a new component or domain](#building-for-different-models)
- [How to verify the new conditional compilation feature](#building-for-devices-with-different-isa)


## Introduction of conditional compilation macro

There are several macros to help enable conditional compilation(CC) feature for new component, and in each CC mode(SELECTIVE_BUILD_ANALYZER and SELECTIVE_BUILD) they must be defined with same name but difference macro content definition. Developer can apply or reference these macros for own component's conditional compilation enablement. You can find these macros in [selective_buld.h](https://github.com/openvinotoolkit/openvino/blob/master/src/common/conditional_compilation/include/openvino/cc/selective_build.h) and [itt.hpp](https://github.com/openvinotoolkit/openvino/blob/master/src/common/conditional_compilation/include/openvino/cc/pass/itt.hpp).


|  Macros | SELECTIVE_BUILD_ANALYZER | SELECTIVE_BUILD | Non Conditional Compilation |
|---------|--------------------------| ----------------|-----------------------------|
| OV_CC_DOMAINS | defines CC domains |  empty | empty  |
| OV_SCOPE | annotate code region | exclude inactive code region | empty |
| OV_SWITCH | annotate match template parameter | exclude inactive template class instance | match template parameter  |
| MATCHER_SCOPE | defines a string for matcher name | exclude inactive code region   | empty   |
| RUN_ON_FUNCTION_SCOPE | annotate code region | exclude inactive code region | empty |
| RUN_ON_MODEL_SCOPE | annotate code region | exclude inactive code region | empty |

### 1. Macro in SELECTIVE_BUILD_ANALYZER mode
#### OV_CC_DOMAINS
It defines domains for conditional compilation, in which contains 3 macros to define detail domains(simple scope, switch/cases and factories). The implement code is:

https://github.com/openvinotoolkit/openvino/blob/713eb9683f67a5eb07374e96b7bbbfcd971ca69e/src/common/conditional_compilation/include/openvino/cc/selective_build.h#L113-L116

Notice: macro `OV_PP_CAT` will connect 2 symbols together, and macro `OV_ITT_DOMAIN` is used to declare domain with a given name for ITT profiling. 

#### OV_SCOPE
It will leverage `OV_ITT_SCOPED_TASK` to help annotate code region until scope exit, which will be profiled whether need to set to be active code region or not, the active code region will generate a macro definition with prefix of “SIMPLE_” in the header file to make it can be built in `SELECTIVE_BUILD` mode, while the inactive code region will be excluded in the following build.

https://github.com/openvinotoolkit/openvino/blob/34b76584f724ad07ed27c8ea27777ac46df92c23/src/common/conditional_compilation/include/openvino/cc/selective_build.h#L159-L160


#### OV_SWITCH
It is mainly used for template class to verify specified input parameter whether match template parameter types. It will check each template parameter types (Cases) and return true if is matched or return false if not match any one. If matched, it will be labeled as active code region. The Cases can be defined by macro **OV_CASE**, which is a `case_wrapper` structure to store data type and its value. Most of case, `OV_SWITCH` will combine `OV_CASE` to work together, there is a simple example:

https://github.com/openvinotoolkit/openvino/blob/34b76584f724ad07ed27c8ea27777ac46df92c23/src/plugins/intel_cpu/src/nodes/one_hot.cpp#L158-L161

And there is another macro named **OV_CASE2**, which can support 2 pairs of type and value.

#### MATCHER_SCOPE
It defines a string to represent the matcher name in this region.

https://github.com/openvinotoolkit/openvino/blob/34b76584f724ad07ed27c8ea27777ac46df92c23/src/common/conditional_compilation/include/openvino/cc/pass/itt.hpp#L21-L21


#### RUN_ON_FUNCTION_SCOPE and RUN_ON_MODEL_SCOPE
They are very similar except for different region suffix, one is run on function, another is run on model. Both them leverage OV_SCOPE with ov_pass as its module name.

https://github.com/openvinotoolkit/openvino/blob/34b76584f724ad07ed27c8ea27777ac46df92c23/src/common/conditional_compilation/include/openvino/cc/pass/itt.hpp#L20-L22


### 2. Macro in SELECTIVE_BUILD mode

#### OV_CC_DOMAINS
It is empty in this mode.

#### OV_CC_SCOPE_IS_ENABLED
It checks whether the condition is true or false, it always is embedded in **OV_SCOPE** or **MATCHER_SCOPE** to decide whether the following code region need be excluded out of final binaries.

#### OV_SCOPE
It will check whether the code region in this module is active or inactive by the macros of header file generated in previous analyzed mode. The generated macros name format is <module>_<region>, in which module name is passed by the parameters of OV_SCOPE. 

https://github.com/openvinotoolkit/openvino/blob/34b76584f724ad07ed27c8ea27777ac46df92c23/src/common/conditional_compilation/include/openvino/cc/selective_build.h#L181-L183


#### OV_SWITCH
It will decide whether the template class parameter match operation need be excluded from final binaries. If the match operation is false in analyzed mode, the code region will be excluded in SELECTIVE_BUILD mode.

#### OV_CASE and OV_CASE2
They are used with OV_SWITCH to provide match cases.

#### MATCHER_SCOPE
It defines a string to represent the matcher name in this region, and then decide whether the code region is inactivated and excluded from final binaries, according to the OV_SCOPE result in analyzed mode.

https://github.com/openvinotoolkit/openvino/blob/34b76584f724ad07ed27c8ea27777ac46df92c23/src/common/conditional_compilation/include/openvino/cc/pass/itt.hpp#L30-L33


#### RUN_ON_FUNCTION_SCOPE and RUN_ON_MODEL_SCOPE
They are very similar, and functionality is same with MATCHER_SCOPE plus throw error.

### 3. Non Conditional Compilation
It means that conditional compilation is disabled.

#### OV_CC_DOMAINS
It is empty in this mode

#### OV_SCOPE
It is empty in this mode

#### OV_SCOPE and OV_CASE
They will do match operation without itt scope task.

#### MATCHER_SCOPE
It is empty.

#### RUN_ON_FUNCTION_SCOPE and RUN_ON_MODEL_SCOPE
They are empty.


## Enable conditional complication for a new component or domain
### step 1: requirement analyze
Analyze which code region need conditional complication and figure out requirement for conditional complication.

### step 2: create itt.hpp if possible
Create itt.hpp file in the target component directory, in which you need define some particular macro needed by conditional compilation if existed macro cannot meet requirement. An example [itt.hpp](https://github.com/openvinotoolkit/openvino/blob/master/src/common/transformations/src/itt.hpp).

- Add macro into itt.hpp to define new CC domain:
```
    #include <openvino/cc/selective_build.h>
    #include <openvino/itt.hpp>
    #include <openvino/cc/ngraph/itt.hpp>

    namespace ov{
    namespace <your_component> {
    namespace itt {
    namespace domains {
        OV_ITT_DOMAIN(your_domain_name);
    }
    }
    }
    }

    OV_CC_DOMAINS(your_domain_name);
```
    
- Define new macro according to the actual requirement by target component
  The same macro should be defined in the 3 modes separately, below is an example:

```
    #if defined(SELECTIVE_BUILD_ANALYZER)

    #define MY_SCOPE(region) OV_SCOPE(my_op, region)

    #elif defined(SELECTIVE_BUILD)

    #define MY_SCOPE(region) MATCHER_SCOPE_(my_op, region)

    #else

    #define MY_SCOPE(region)

    #endif
```

### step 3: apply CC macro
Apply the CC macro into your code region that need conditional compilation.
An example:

https://github.com/openvinotoolkit/openvino/blob/34b76584f724ad07ed27c8ea27777ac46df92c23/src/common/transformations/src/ngraph_ops/augru_cell.cpp#L76-L85

Then code region in ov::op::internal::AUGRUCell::clone_with_new_inputs() is implement conditional compilation. If in the SELECTIVE_BUILD_ANALYZER mode, it was not called, then in SELECTIVE_BUILD Mode, the code region in this function will be stripped out.


## How to verify the new conditional compilation feature

#### Build OpenVINO in SELECTIVE_BUILD_ANALYZER mode
```
    dir=`pwd`
    mkdir build && cd build
    cmake \
            -DCMAKE_BUILD_TYPE=Release \
            -DENABLE_PROFILING_ITT=ON \
            -DSELECTIVE_BUILD=COLLECT \
            ..
    make -j12
```

#### Generate and collect analyzed data
```
    cd thirdparty/itt_collector
    make
    cd $dir
    mkdir -p $dir/cc_data
    cd $dir/bin/intel64/Release
    python ../../../thirdparty/itt_collector/runtool/sea_runtool.py --bindir $OPENVINO_HOME/bin/intel64/Release -o  <cc_data_dir>/data ! ./benchmark_app -niter 1 -nireq 1 -m <your_model.xml> -d CPU
    cd $dir
```

#### Build OpenVINO in SELECTIVE_BUILD mode
```
    cmake \
            -DCMAKE_BUILD_TYPE=Release \
            -DENABLE_LTO=ON \
            -DSELECTIVE_BUILD=ON \
            -DSELECTIVE_BUILD_STAT=$OPENVINO_HOME/cc_data/*.csv \
            -DENABLE_PROFILING_ITT=OFF 
            ..
    make -j8
```
    Note: It will benefit more if build OpenVINO static library with conditional complication, you can add option of `-DBUILD_SHARED_LIBS=OFF`.

#### Run test sample 
Run test samples built in SELECTIVE_BUILD and check whether there will be any error reported.
```
    ./benchmark_app -m <your_model.xml> -d CPU
``` 

#### Check the binaries size benefit.
Get the binaries size and compare with master branch result(need do as step 1 to 4), check whether the binaries is smaller than master.
