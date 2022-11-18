# How to enable conditional compilation for new component

## Contents

- [Introduce conditional compilation macro](#Introduction-of-conditional-compilation-macro)
- [Enable conditional complication for a new component or domain](#building-for-different-models)
- [How to verify the new conditional compilation feature](#building-for-devices-with-different-isa)


## Introduce conditional compilation macro

There are several macros to help enable conditional compilation(CC) feature for new component, and in each CC mode(SELECTIVE_BUILD_ANALYZER and SELECTIVE_BUILD) they must be defined with same name but difference macro content. Developer can apply or reference these macros for own component's conditional compilation enablement. You can find these macros in [selective_buld.h](https://github.com/openvinotoolkit/openvino/blob/master/src/common/conditional_compilation/include/openvino/cc/selective_build.h) and [itt.hpp](https://github.com/openvinotoolkit/openvino/blob/master/src/common/conditional_compilation/include/openvino/cc/pass/itt.hpp).


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
It defines domains for conditional compilation, in which contains 3 macros to define detail domains(simple scope, switch/cases and factories). The implement code is:

https://github.com/openvinotoolkit/openvino/blob/713eb9683f67a5eb07374e96b7bbbfcd971ca69e/src/common/conditional_compilation/include/openvino/cc/selective_build.h#L113-L116

Notice: macro `OV_PP_CAT` will connect 2 symbols together, and macro `OV_ITT_DOMAIN` is used to declare domain with a given name for ITT profiling. 

#### OV_SCOPE
It will leverage `OV_ITT_SCOPED_TASK` to help annotate code region until scope exit, which will be profiled whether need to set to be active code region or not, the active code region will generate a macro definition with prefix of “SIMPLE_” in the header file to make it can be built in `SELECTIVE_BUILD` mode, while the inactive code region will be excluded in the following build.

https://github.com/openvinotoolkit/openvino/blob/34b76584f724ad07ed27c8ea27777ac46df92c23/src/common/conditional_compilation/include/openvino/cc/selective_build.h#L159-L160


#### OV_SWITCH
It is mainly used for template class to verify specified input parameter whether match template parameter types. It will check each template parameter types (`Cases`) and return true if is matched or return false if not match any one. If matched, it will be labeled as active code region. The `Cases` can be defined by macro *OV_CASE*, which is a `case_wrapper` structure to store data type and its value. Most of case, `OV_SWITCH` will combine `OV_CASE` to work together, there is a simple example:

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
It will check whether the code region in this module is active or inactive by the macros in header file(`conditional_compilation_gen.h`) generated in previous analyzed mode. The generated macros name format is <module>_<region>, in which module name is passed by the parameters of OV_SCOPE.

https://github.com/openvinotoolkit/openvino/blob/34b76584f724ad07ed27c8ea27777ac46df92c23/src/common/conditional_compilation/include/openvino/cc/selective_build.h#L181-L183

There is an example of `conditional_compilation_gen.h`:
```
#define ov_pass_FixRtInfo_run_on_function 1
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
Create itt.hpp file in the target component directory, in which you need define some particular macro needed by conditional compilation if existed macro cannot meet requirement. An example [itt.hpp](https://github.com/openvinotoolkit/openvino/blob/master/src/core/src/itt.hpp).

- Add macro into itt.hpp to define new CC domain:

https://github.com/openvinotoolkit/openvino/blob/be277ab9772de827739ccf960bf7cebf1c6f06b0/src/core/src/itt.hpp#L12-L28
    
- Define new macro according to the actual requirement by target component
  The same macro should be defined in the 3 modes separately, below is an example:

https://github.com/openvinotoolkit/openvino/blob/be277ab9772de827739ccf960bf7cebf1c6f06b0/src/core/src/itt.hpp#L34-L57


### step 3: apply CC macros
Apply the CC macros into your code region that need conditional compilation.
An example:

https://github.com/openvinotoolkit/openvino/blob/807279276b78e1fdf9e1e0babd427e1e8dd9a07b/src/core/src/op/abs.cpp#L19-L23

Then code region in ov::op::v0::Abs::clone_with_new_inputs() is implement conditional compilation.
In `SELECTIVE_BUILD=COLLECT` stage, if it was not called, then in `SELECTIVE_BUILD=ON` stage, the code region in this function will be stripped out.


## How to verify the new conditional compilation feature

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
    Note: It will benefit more if build OpenVINO static library with conditional complication, you can add option of `-DBUILD_SHARED_LIBS=OFF`.

#### Run test sample 
Run test samples built in SELECTIVE_BUILD and check whether there will be any error reported.
```
    ./benchmark_app -m <your_model.xml> -d CPU
``` 

#### Check the binaries size benefit.
Get the binaries size and compare with master branch result(need do as step 1 to 4), check whether the binaries is smaller than master.