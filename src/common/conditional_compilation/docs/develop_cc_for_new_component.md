# How to enable conditional compilation for new component

## Contents

- [Introduce conditional compilation macro](#introduce-conditional-compilation-macro)
- [Enable conditional compilation for a new component or domain](#enable-conditional-compilation-for-a-new-component-or-domain)
- [Verify the new conditional compilation feature](#verify-the-new-conditional-compilation-feature)

## Introduce conditional compilation macro

There are several macros to help enable the conditional compilation (CC) feature for a new component. They must be defined with the same name but different macro content in each CC mode (SELECTIVE BUILD ANALYZER and SELECTIVE BUILD). You can apply or reference these macros to enable conditional compilation for your own component. You can find these macros in [selective_buld.h](https://github.com/openvinotoolkit/openvino/blob/master/src/common/conditional_compilation/include/openvino/cc/selective_build.h) and [itt.hpp](https://github.com/openvinotoolkit/openvino/blob/master/src/common/conditional_compilation/include/openvino/cc/pass/itt.hpp).


|  Macros | SELECTIVE_BUILD_ANALYZER | SELECTIVE_BUILD | Off Conditional Compilation |
|---------|--------------------------| ----------------|-----------------------------|
| OV_CC_DOMAINS | defines CC domains |  empty | empty  |
| OV_SCOPE | annotate code region | exclude inactive code region | empty |
| OV_SWITCH | annotate match template parameter | exclude inactive template class instance | match template parameter  |
| MATCHER_SCOPE | defines a string for matcher name | exclude inactive code region   | empty   |
| RUN_ON_FUNCTION_SCOPE | annotate code region | exclude inactive code region | empty |
| RUN_ON_MODEL_SCOPE | annotate code region | exclude inactive code region | empty |

### 1. Macro in SELECTIVE_BUILD_ANALYZER mode
#### OV_CC_DOMAINS

It defines domains for conditional compilation, which contains three macros to define detail domains (simple scope, switch/cases, and factories). 

[Open the code](https://github.com/openvinotoolkit/openvino/blob/713eb9683f67a5eb07374e96b7bbbfcd971ca69e/src/common/conditional_compilation/include/openvino/cc/selective_build.h#L113-L116)

> **NOTE**: macro `OV_PP_CAT` connects two symbols together, and macro `OV_ITT_DOMAIN` is used to declare a domain with a given name for ITT profiling. 

#### OV_SCOPE
It leverages `OV_ITT_SCOPED_TASK` to help annotate a code region until the scope exit, which will be profiled depending on whether the active code area needs to be set or not. The active code region will generate a macro definition with the prefix “SIMPLE_” in the header file so that it can be built in the `SELECTIVE_BUILD` mode, while the inactive code region will be excluded in the following build.

[Open the code](https://github.com/openvinotoolkit/openvino/blob/34b76584f724ad07ed27c8ea27777ac46df92c23/src/common/conditional_compilation/include/openvino/cc/selective_build.h#L159-L160)


#### OV_SWITCH
It is mainly used for the template class to verify whether the specified input parameter matches template parameter types. It checks each template parameter type (`Cases`) and returns true if it matches or false if there is no match. If matched, it is labeled as an active code region. The `Cases` can be defined by macro *OV_CASE*, which is a `case_wrapper` structure to store data type and its value. In most cases, `OV_SWITCH` combines `OV_CASE` to work together:

[Open the code](https://github.com/openvinotoolkit/openvino/blob/34b76584f724ad07ed27c8ea27777ac46df92c23/src/plugins/intel_cpu/src/nodes/one_hot.cpp#L158-L161)

There is another macro named **OV_CASE2**, which can support two pairs of type and value.

#### MATCHER_SCOPE
It defines a string to represent the matcher name in this region.

[Open the code](https://github.com/openvinotoolkit/openvino/blob/34b76584f724ad07ed27c8ea27777ac46df92c23/src/common/conditional_compilation/include/openvino/cc/pass/itt.hpp#L21-L21)


#### RUN_ON_FUNCTION_SCOPE and RUN_ON_MODEL_SCOPE
They are very similar except for different region suffix. One is run on function, and another is run on model. Both of them leverage OV_SCOPE with ov_pass as its module name.

[Open the code](https://github.com/openvinotoolkit/openvino/blob/34b76584f724ad07ed27c8ea27777ac46df92c23/src/common/conditional_compilation/include/openvino/cc/pass/itt.hpp#L20-L22)


### 2. Macro in SELECTIVE_BUILD mode

#### OV_CC_DOMAINS
Empty in this mode.

#### OV_CC_SCOPE_IS_ENABLED
It checks whether the condition is true or false. It is always embedded in **OV_SCOPE** or **MATCHER_SCOPE** to decide whether the following code region needs to be excluded from the final binaries.

#### OV_SCOPE
It checks whether the code region in this module is active or inactive by the macros in the header file(`conditional_compilation_gen.h`) generated in the previous analyzed mode. The generated macros name format is <module>_<region>, in which the module name is passed by the parameters of OV_SCOPE.

[Open the code](https://github.com/openvinotoolkit/openvino/blob/34b76584f724ad07ed27c8ea27777ac46df92c23/src/common/conditional_compilation/include/openvino/cc/selective_build.h#L181-L183)

There is an example of `conditional_compilation_gen.h`:
```
#define ov_pass_GraphRewrite_run_on_model 1
#define ov_pass_InitNodeInfo_run_on_function 1
#define ov_pass_ConstantFolding_run_on_model 1
#define ov_pass_SubtractFusion 1
#define ov_pass_SharedShapeOf_run_on_function 1
#define ov_pass_SimplifyShapeOfSubGraph_run_on_function 1
#define ov_pass_PropagateNMSPath 1
#define ov_pass_UselessStridedSliceEraser_run_on_function 1
#define ov_pass_SharedStridedSliceEraser_run_on_function 1
#define ov_pass_GroupedStridedSliceOptimizer_run_on_function 1
#define ov_pass_StridedSliceOptimization_run_on_function 1
#define ov_pass_EliminateConvert 1
#define ov_pass_EliminateEltwise 1
#define ov_pass_Unsupported 1
#define ov_pass_PassThrough 1
#define ov_pass_Binary 1
#define ov_pass_ConvertPassThrough 1
#define ov_pass_BackwardGraphRewrite_run_on_model 1
#define ov_pass_Constant_run_on_function 1
```

#### OV_SWITCH
It decides whether the template class parameter match operation needs to be excluded from the final binaries. If the match operation is false in analyzed mode, the code region will be excluded in SELECTIVE_BUILD mode.

#### OV_CASE and OV_CASE2
They are used with OV_SWITCH to provide match cases.

#### MATCHER_SCOPE
It defines a string to represent the matcher name in this region. Then decides whether the code region is inactivated and excluded from the final binaries according to the OV_SCOPE result in analyzed mode.

[Open the code](https://github.com/openvinotoolkit/openvino/blob/34b76584f724ad07ed27c8ea27777ac46df92c23/src/common/conditional_compilation/include/openvino/cc/pass/itt.hpp#L30-L33)

#### RUN_ON_FUNCTION_SCOPE and RUN_ON_MODEL_SCOPE
They are very similar, and functionality are the same with MATCHER_SCOPE plus throw error.

### 3. Non Conditional Compilation
It means that conditional compilation is disabled.

#### OV_CC_DOMAINS
Empty in this mode.

#### OV_SCOPE
Empty in this mode.

#### OV_SCOPE and OV_CASE
They do the match operation without the itt scope task.

#### MATCHER_SCOPE
Empty.

#### RUN_ON_FUNCTION_SCOPE and RUN_ON_MODEL_SCOPE
Empty.

## Enable conditional compilation for a new component or domain

### Step 1: analyze requirements

Analyze which code region need conditional compilation and figure out requirements for conditional compilation.

### Step 2: create itt.hpp if possible

Create itt.hpp file in the target component directory, in which you need to define a particular macro needed by the conditional compilation, if the existing macro cannot meet the requirement. An example [itt.hpp](https://github.com/openvinotoolkit/openvino/blob/master/src/core/src/itt.hpp).

- Add macro into itt.hpp to define a new CC domain:

    [Open the code](https://github.com/openvinotoolkit/openvino/blob/be277ab9772de827739ccf960bf7cebf1c6f06b0/src/core/src/itt.hpp#L12-L28)
        
- Define a new macro according to the requirement of the target component. The same macro should be defined in the three modes separately.

    [Open the code example](https://github.com/openvinotoolkit/openvino/blob/be277ab9772de827739ccf960bf7cebf1c6f06b0/src/core/src/itt.hpp#L34-L57)

### Step 3: apply CC macros

Apply the CC macros for your code region that needs conditional compilation.

[Open the code example](https://github.com/openvinotoolkit/openvino/blob/807279276b78e1fdf9e1e0babd427e1e8dd9a07b/src/core/src/op/abs.cpp#L19-L23)

Then the code region in ov::op::v0::Abs::clone_with_new_inputs() is implemented with conditional compilation.
In the `SELECTIVE_BUILD=COLLECT` stage, or if it was not called, in the `SELECTIVE_BUILD=ON` stage, the code region in this function will be stripped out.

## Verify the new conditional compilation feature

#### Build OpenVINO in SELECTIVE_BUILD_ANALYZER mode

```
    cmake \
            -DCMAKE_BUILD_TYPE=Release \
            -DENABLE_PROFILING_ITT=ON \
            -DSELECTIVE_BUILD=COLLECT \
            -B build \
            -S .
    cmake --build build -j `nproc`
```

#### Generate and collect analyzed data

```
    cmake --build build --target sea_itt_lib
    mkdir -p cc_data
    python thirdparty/itt_collector/runtool/sea_runtool.py --bindir ./bin/intel64/Release -o ./cc_data ! ./bin/intel64/Release/benchmark_app -niter 1 -nireq 1 -m <your_model.xml> -d CPU
```

#### Build OpenVINO in SELECTIVE_BUILD mode
```
    cmake \
            -DCMAKE_BUILD_TYPE=Release \
            -DENABLE_LTO=ON \
            -DSELECTIVE_BUILD=ON \
            -DSELECTIVE_BUILD_STAT=$OPENVINO_HOME/cc_data/*.csv \
            -DENABLE_PROFILING_ITT=OFF \
            -B build \
            -S .
    cmake --build build -j `nproc`
```
> **NOTE**: It is more beneficial to build OpenVINO static library with conditional compilation. For that, add the `-DBUILD_SHARED_LIBS=OFF` option.

#### Run a test sample 

Run test samples built in SELECTIVE_BUILD and check for any errors.
```
    ./benchmark_app -m <your_model.xml> -d CPU
``` 

#### Check the binaries size benefit

Get the binaries size and compare with the master branch result (need to do as step 1 to 4), check whether the binaries are smaller than in the master.