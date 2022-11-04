# How to enable conditional compilation for new component

There are several macros to help enable conditional compilation(CC) feature for new component, and in each CC mode(SELECTIVE_BUILD_ANALYZER and SELECTIVE_BUILD) they must be defined with same name but difference macro content definition. Developer can apply or reference these macros for own component's conditional compilation enablement. You can find these macros in [selective_buld.h](https://github.com/openvinotoolkit/openvino/blob/master/src/common/conditional_compilation/include/openvino/cc/selective_build.h) and [itt.hpp](https://github.com/openvinotoolkit/openvino/blob/master/src/common/conditional_compilation/include/openvino/cc/pass/itt.hpp).

## Introduction of conditional compilation macro

### 1. Macro in SELECTIVE_BUILD_ANALYZER mode
#### OV_CC_DOMAINS
It defines domains for conditional compilation, in which contains 3 macros to define detail domains(simple scope, switch/cases and factories). The implement code is:
```ruby
    #  define OV_CC_DOMAINS(Module)                                                                   \
       OV_ITT_DOMAIN(OV_PP_CAT(SIMPLE_, Module));  /* Domain for simple scope surrounded by ifdefs */ \
       OV_ITT_DOMAIN(OV_PP_CAT(SWITCH_, Module));  /* Domain for switch/cases */                      \
       OV_ITT_DOMAIN(OV_PP_CAT(FACTORY_, Module)); /* Domain for factories */
```
Notice: macro `OV_PP_CAT` will connect 2 symbols together, and macro `OV_ITT_DOMAIN` is used to declare domain with a given name for ITT profiling. 

#### OV_SCOPE
It will leverage `OV_ITT_SCOPED_TASK` to help annotate code region until scope exit, which will be profiled whether need to set to be active code region or not, the active code region will generate a macro definition with prefix of “SIMPLE_” in the header file to make it can be built in `SELECTIVE_BUILD` mode, while the inactive code region will be excluded in the following build.
```ruby
    # define OV_SCOPE(Module, region) OV_ITT_SCOPED_TASK(OV_PP_CAT(SIMPLE_, Module), OV_PP_TOSTRING(region));
```

#### OV_SWITCH
It is mainly used for template class to verify specified input parameter whether match template parameter types. It will check each template parameter types (Cases) and return true if is matched or return false if not match any one. If matched, it will be labeled as active code region. The Cases can be defined by macro **OV_CASE**, which is a `case_wrapper` structure to store data type and its value. Most of case, `OV_SWITCH` will combine `OV_CASE` to work together, there is a simple example:
```ruby
   OV_SWITCH(intel_cpu, OneHotExecute, ctx, output_precision.size(),
              OV_CASE(sizeof(uint32_t), uint32_t),
              OV_CASE(sizeof(uint16_t), uint16_t),
              OV_CASE(sizeof(uint8_t), uint8_t))
```
And there is another macro named **OV_CASE2**, which can support 2 pairs of type and value.

#### MATCHER_SCOPE
It defines a string to represent the matcher name in this region.
```ruby
    # define MATCHER_SCOPE(region)         const std::string matcher_name(OV_PP_TOSTRING(region))
```

#### RUN_ON_FUNCTION_SCOPE and RUN_ON_MODEL_SCOPE
They are very similar except for different region suffix, one is run on function, another is run on model. Both them leverage OV_SCOPE with ov_pass as its module name.
```ruby
    # define RUN_ON_FUNCTION_SCOPE(region) OV_SCOPE(ov_pass, OV_PP_CAT(region, _run_on_function))
    # define RUN_ON_MODEL_SCOPE(region)    OV_SCOPE(ov_pass, OV_PP_CAT(region, _run_on_model))
```


### 2. Macro in SELECTIVE_BUILD mode

#### OV_CC_DOMAINS
It is empty in this mode.

#### OV_CC_SCOPE_IS_ENABLED
It checks whether the condition is true or false, it always is embedded in **OV_SCOPE** or **MATCHER_SCOPE** to decide whether the following code region need be excluded out of final binaries.

#### OV_SCOPE
It will check whether the code region in this module is active or inactive by the macros of header file generated in previous analyzed mode. The generated macros name format is <module>_<region>, in which module name is passed by the parameters of OV_SCOPE. 
```ruby
    # define OV_SCOPE(Module, region)                                                   \
        for (bool ovCCScopeIsEnabled = OV_PP_IS_ENABLED(OV_PP_CAT3(Module, _, region)); ovCCScopeIsEnabled;  ovCCScopeIsEnabled = false)
```

#### OV_SWITCH
It will decide whether the template class parameter match operation need be excluded from final binaries. If the match operation is false in analyzed mode, the code region will be excluded in SELECTIVE_BUILD mode.

#### OV_CASE and OV_CASE2
They are used with OV_SWITCH to provide match cases.

#### MATCHER_SCOPE
It defines a string to represent the matcher name in this region, and then decide whether the code region is inactivated and excluded from final binaries, according to the OV_SCOPE result in analyzed mode.
```ruby
    # define MATCHER_SCOPE(region)                                        \
        const std::string matcher_name(OV_PP_TOSTRING(region));          \
        if (OV_CC_SCOPE_IS_ENABLED(OV_PP_CAT3(ov_pass, _, region)) == 0) \
        return
```

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
```ruby
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

```ruby
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
```ruby
    shared_ptr<ov::Node> ov::op::internal::AUGRUCell::clone_with_new_inputs(const OutputVector& new_args) const {
        My_SCOPE(internal_AUGRUCell_clone_with_new_inputs);
        check_new_args_count(this, new_args);
        return make_shared<AUGRUCell>(new_args.at(0),
                                  new_args.at(1),
                                  new_args.at(2),
                                  new_args.at(3),
                                  new_args.at(4),
                                  new_args.at(5),
                                  get_hidden_size());
    }
```
   Then code region in ov::op::internal::AUGRUCell::clone_with_new_inputs() is implement conditional compilation. If in the SELECTIVE_BUILD_ANALYZER mode it was not called, then in SELECTIVE_BUILD Mode, the code region in this function will be stripped out. 



## How to verify the new conditional compilation feature

#### Build OpenVINO in SELECTIVE_BUILD_ANALYZER mode
```ruby
    cmake \
            -DCMAKE_BUILD_TYPE=Release \
            -DENABLE_PROFILING_ITT=ON \
            -DSELECTIVE_BUILD=COLLECT \
            ..
    make -j12
```

#### Generate and collect analyzed data
```ruby
    dir=`pwd`
    cd build-x86_64/thirdparty/itt_collector
    make
    cd $dir
    mkdir -p $dir/cc_data
    cd $dir/bin/intel64/Release
    python3.8 ../../../thirdparty/itt_collector/runtool/sea_runtool.py --bindir $OPENVINO_HOME/bin/intel64/Release -o  <cc_data_dir>/data ! ./benchmark_app -niter 1 -nireq 1 -m <your_model.xml> -d CPU
    cd $dir
```

#### Build OpenVINO in SELECTIVE_BUILD mode
```ruby
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
```ruby
    ./benchmark_app -m <your_model.xml> -d CPU
``` 

#### Check the binaries size benefit.
Get the binaries size and compare with master branch result(need do as step 1 to 4), check whether the binaries is smaller than master, in general 100~500 KB.
