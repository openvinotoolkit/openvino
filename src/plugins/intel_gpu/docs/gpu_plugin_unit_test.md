# GPU plugin unit test

GPU plugin has two type tests: first one is functional tests and second one is unit tests.

- The functional test is testing single layer, behavior, sub graph and low precision transformation on inference engine level for various layout and data types such as fp16 and fp32.
- The unit test is testing cldnn primitive and core type modules on GPU plugin level. Unlike functional test, it is possible to test by explicitly specifying the format of the input such as `bfyx` or `b_fs_yx_fsv16`. This documentation is about this type of test.

# Structure of unit test

Intel GPU unit test (aka clDNN unit test) is a set of unit tests each of which is for testing all primitives, fusions and fundamental core types of GPU plugin. 
There are 4 sub categories of unit tests as below.

```bash
openvino/src/plugins/intel_gpu/tests	- root of Intel GPU unit test
|── fusions
|── module_tests 				
|── test_cases
└── test_utils
```

- ### fusions
  - Fusion is an algorithm that fuse several operations into one optimized operation. For example, two nodes of `conv -> relu` may be fused into single node of `conv`.
  - Fusion unit tests checks whether the fusion is done as expected.
  - fusion_test_common.cpp
     - The base class for fusing test, i.e., [BaseFusingTest](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_gpu/tests/fusions/fusion_test_common.hpp#L19), is implemented here. It tests whether the fusing is successful or not by comparing the execution results of the two networks, one is the fused network, the other is non fused network for same topology.
       - [BaseFusingTest](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_gpu/tests/fusions/fusion_test_common.hpp#L19) has an important method called *`compare()`*. 
       - *`compare()`* method has the following three tasks
            - Execute two networks (fused network and not fused network)
            - Compare the actual  number of executed primitives with the expected number of executed primitives in test params
            - Compare the results between fused network and non fused network
  - eltwise_fusing_test.cpp
       - Check whether or not eltwise is fused to other primitives as expected
  - [primitive_name]_fusion_test.cpp
       - Check that nodes such as eltwise or activation are fusing to the [primitive_name] as expected
  - The detail of how to add each instance is described [below](#fusions-1).

- ### test_cases
  - It is mainly checking that cldnn primitives and topology creation are working as designed
  - It also checks configurations for OpenCL functionalities such as cl_cache, cl_mem allocation and cl_command_queue modes

- ### module_tests 
  - Unit tests for fundamental core modules such as ocl_user_events, format, layout, and usm memory
    - Check ocl_user_event is working as expected
    - Check all format is converted to the string and trait
    - Check various layouts are created as expected
    - Check usm_host and  usm device memory buffer creation and read/write functionality

- ### test_utils
  - Defined base functions of unit test such as *`get_test_engine()`* which returns `cldnn::engine`
  - Utility functions such as Float16, random_gen and uniform_quantized_real_distribution


# How to run unit tests

## Build unit test

1. Turn on `ENABLE_TESTS` and `ENABLE_CLDNN_TESTS` in cmake option

   ```bash
   cmake -DCMAKE_BUILD_TYPE=Release \
       -DENABLE_TESTS=ON \
       -DENABLE_CLDNN_TESTS=ON \
       -DENABLE_CLDNN=ON ..
   ```

2. Build

   ```bash
   make clDNN_unit_tests
   ```

3. You can find _`clDNN_unit_tests64`_ in bin directory after build



## Run unit test

You can run _`clDNN_unit_tests64`_ in bin directory which is the output of openvino build

If you want to run specific unit test, you can use gtest_filter option as follows:

```
./clDNN_unit_tests64 --gtest_filter='*filter_name*'
```

Then, you can get the result like this

```bash
openvino/bin/intel64/Release$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD
openvino/bin/intel64/Release$ ./clDNN_unit_tests64 --gtest_filter='*fusings_gpu/conv_fp32_reorder_fsv16_to_bfyx.basic/0*'
Running main() from /home/openvino/thirdparty/gtest/gtest/googletest/src/gtest_main.cc
Note: Google Test filter = *fusings_gpu/conv_fp32_reorder_fsv16_to_bfyx.basic/0*
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from fusings_gpu/conv_fp32_reorder_fsv16_to_bfyx
[ RUN      ] fusings_gpu/conv_fp32_reorder_fsv16_to_bfyx.basic/0
[       OK ] fusings_gpu/conv_fp32_reorder_fsv16_to_bfyx.basic/0 (84 ms)
[----------] 1 test from fusings_gpu/conv_fp32_reorder_fsv16_to_bfyx (84 ms total)
[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (85 ms total)
[  PASSED  ] 1 test.
```


# How to create new test case

## TEST and TEST_P (GoogleTest macros)

GPU unit tests are using 2 types of test macros(**TEST** and **TEST_P**)  in  [GoogleTest (aka gtest)](https://google.github.io/googletest/)

- ### **TEST**
  - **TEST** is the simple test case macro.
  - To make test-case using **TEST**,  define an individual test named *`TestName`* in the test suite *`TestSuiteName`*

    ```
    TEST(TestSuiteName, TestName) {
      ... test body ...
    }
    ```
  - The test body can be any code under test. To determine the outcomes within the test body, use assertion such as *`EXPECT_EQ`* and *`ASSERT_NE`*.
 
- ### **TEST_P**
  - **TEST_P** is used to set test case using test parameter sets
  - To make test-case using **TEST_P**, define an individual value-parameterized test named *`TestName`* that uses the test fixture class *`TestFixtureName`* which is the test suite name

    ```
    TEST_P(TestFixtureName, TestName) {
      ... statements ...
    }
    ```
  - Then, instantiates the value-parameterized test suite *`TestSuiteName`* which is defined defined with **TEST_P**
    ```c++
    INSTANTIATE_TEST_SUITE_P(InstantiationName,TestSuiteName,param_generator)
    ```


## module_test and test_cases

- module_test and test_cases are testing GPU plugin using both **TEST_P** and **TEST**.
- Please refer to [the fusion test](#fusions-1) for the test case based on **TEST_P**
- **TEST** checks the test result by comparing the execution results with expected values after running network created from the target topology to check.
  - It is important to generate test input and expected output result in **TEST**
  - You can create input data and expected output data using the 3 following ways:
    - Generate simple input data and calculate the expected output data from input data manually like [basic_deformable_convolution_def_group1_2](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_gpu/tests/test_cases/convolution_gpu_test.cpp#L254)
    - Generate random input and get the expected output using reference function which is made in the test codes like [mvn_test_across_channels_outside_sqrt_bfyx](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_gpu/tests/test_cases/mvn_gpu_test.cpp#L108)
    - Generate random input and get the expected output from another reference kernel which is existed in cldnn kernels like [mvn_random_test_bsv32](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_gpu/tests/test_cases/mvn_gpu_test.cpp#L793)

- When you allocate input data, please keep in mind that the layout order in *`engine.allocation_memory`* is not *`bfyx`* but *`bfxy`*. i.e., example, if input is {1,1,4,5}, the layout should be below

  ```c++
  auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 5, 4 } });
  ```


## fusions

- It is implemented based on **TEST_P** because there are many cases where multiple layouts are tested in the same topology
- If the fusing test class is already existed, you can use it. otherwise, you should make new fusing test class which is inherited [BaseFusingTest](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_gpu/tests/fusions/fusion_test_common.hpp#L19)
  - The new fusing test class should create `execute()` method which creates fused / non fused networks and calls *`compare`* method after setting input
- Create test case using **TEST_P**
  - You can make the desired networks using create_topologies. 
  ![image-20220326010242344](https://user-images.githubusercontent.com/30511605/160246105-36b01baf-1811-40ee-a497-778f23961649.png)

  - For example, if you design the networks like the one above, you can make the test code as follow

    ```c++
    class conv_fp32_multi_eltwise_4_clamp : public ConvFusingTest {};
    TEST_P(conv_fp32_multi_eltwise_4_clamp, basic) {
        if (engine.get_device_info().supports_immad) {
            return;
        }
        auto p = GetParam();
        create_topologies(
            input_layout("input", get_input_layout(p)),
            data("eltwise1_data", get_mem(get_output_layout(p))),
            data("eltwise2_data", get_mem(get_output_layout(p))),
            data("eltwise4_data", get_mem(get_output_layout(p))),
            data("bias", get_mem(get_bias_layout(p))),
            data("weights", get_mem(get_weights_layout(p))),
            convolution("conv_prim", "input", { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
            eltwise("eltwise1_add", "conv_prim", "eltwise1_data", eltwise_mode::sum),
            activation("activation", "eltwise1_add", activation_func::clamp, { 0.5f, 2.5f }),
            eltwise("eltwise2_mul", "activation", "conv_prim", eltwise_mode::prod),
            eltwise("eltwise3_div", "eltwise2_mul", "eltwise2_data", eltwise_mode::prod),
            eltwise("eltwise4_add", "eltwise3_div", "eltwise4_data", eltwise_mode::sum),
            reorder("reorder_bfyx", "eltwise4_add", p.default_format, data_types::f32)
        );
        implementation_desc conv_impl = { format::b_fs_yx_fsv16, "" };
        bo_fused.set_option(build_option::force_implementations({ { "conv_prim", conv_impl } }));
        tolerance = 1e-5f;
        execute(p);
    }
    
    ```

  - If you want to change some node's layout format to specific format, you can change it using *`build_option::force_implementations`*.
    - In the sample codes, *`conv_prim`* is set to *`format::b_fs_yx_fsv16`* by *`build_option::force_implementations`*
- *`tolerance`* is used as to threshold to check whether or not output result are same between fused network and non fused network in *`compare`* function.
- After the test case is implemented, use `INSTANTIATE_TEST_SUITE_P` to set the test suite for each parameter case as follows. 
  - Check all variables in *`convolution_test_params`* to make `CASE_CONV_FP32_2`. 
    - In *`convolution_test_params`*, all tensor, format, and data_types are used in common in all convolution fusing tests. So you can define `CASE_CONV_FP32_2` with all variables except *`expected_fused_primitives`* and *`expected_not_fused_primitives`*

```c++
struct convolution_test_params {
    tensor in_shape;
    tensor out_shape;
    tensor kernel;
    tensor stride;
    tensor pad;
    tensor dilation;
    uint32_t groups;
    data_types data_type;
    format input_format;
    data_types weights_type;
    format weights_format;
    data_types default_type;
    format default_format;
    size_t expected_fused_primitives;
    size_t expected_not_fused_primitives;
};


// in_shape; out_shape; kernel; stride; pad; dilation; groups; data_type; input_format; weights_type; weights_format; default_type; default_format;
#define CASE_CONV_FP32_2 { 1, 16, 4, 5 }, { 1, 32, 2, 3 }, { 1, 1, 3, 3 }, tensor{ 1 }, tensor{ 0 }, tensor{ 1 }, 1, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::os_is_yx_isv16_osv16, data_types::f32, format::bfyx


INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_scale, ::testing::ValuesIn(std::vector<convolution_test_params>{
    convolution_test_params{ CASE_CONV_FP32_2, 2, 3 }, // CASE_CONV_FP32_2, # of fused executed primitives, # of non fused networks
    convolution_test_params{ CASE_CONV_FP32_3, 2, 3 },
}));
```

## See also
 * [OpenVINO™ README](../../../../README.md)
 * [OpenVINO Core Components](../../../README.md)
 * [OpenVINO Plugins](../../README.md)
 * [OpenVINO GPU Plugin](../README.md)
 * [Developer documentation](../../../../docs/dev/index.md)