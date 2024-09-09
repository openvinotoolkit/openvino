OpenVINO Conditional Compilation for Optimal Binary Size
========================================================


Conditional compilation can significantly reduce the binary size of the OpenVINO package by excluding unnecessary components for inference of particular models. This is particularly useful for edge and client deployment scenarios, where reducing the size of application binary is important.

.. important::

    The tradeoff of conditional compilation is that the produced OpenVINO runtime can only run inference for the models and platforms which conditional compilation was applied.


Lean more in the `conditional_compilation_guide <https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/conditional_compilation.md>`__ and `Conditional_compilation_developer_guide <https://github.com/openvinotoolkit/openvino/blob/master/src/common/conditional_compilation/docs/develop_cc_for_new_component.md>`__

There are two steps to reduce binary size of the OpenVINO runtime library with conditional compilation:

- Apply ``SELECTIVE_BUILD=COLLECT`` and ``DENABLE_PROFILING_ITT=ON`` build options to enable analysis mode of conditional compilation to collect statistics data using ``itt``.

- Apply ``SELECTIVE_BUILD=ON`` and ``SELECTIVE_BUILD_STAT=<statistics_data.csv>`` build options to exclude inactive code region with the help of previous statistics data and get the final OpenVINO package.

.. note::

    install ``Python`` to help collect statistics data.


Conditional Compilation for Multiple Models
############################################

Stage 1: collecting statistics information about code usage
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

- Build OpenVINO with ``SELECTIVE_BUILD=COLLECT`` option

  .. code-block:: sh

    git submodule init
    git submodule update
    mkdir build && cd build
    cmake -DENABLE_PROFILING_ITT=ON -DSELECTIVE_BUILD=COLLECT ..
    cmake --build .

- Build ITT collector for code usage analysis

  .. code-block:: sh

    cmake --build . --target sea_itt_lib

- Run the target application under the ITT collector for code usage analysis of each model

  .. code-block:: sh

    python thirdparty/itt_collector/runtool/sea_runtool.py --bindir ${OPENVINO_LIBRARY_DIR} -o ${MY_MODEL_RESULT} ! ./benchmark_app -niter 1 -nireq 1 -m ${MY_MODEL}.xml

    Then, statistics information are generated and stored into .cvs format files under ``{MY_MODEL_RESULT}`` directory.

  .. tip::

    If you want to run other application rather than benchmark_app to get statistics data, please make sure to limit inference request number and iterations to avoid too long profiling time and too large statistics data.


    You can run this `script <https://github.com/openvinotoolkit/openvino/blob/master/src/common/conditional_compilation/scripts/ccheader.py>`__ to get the generated header file from csv files, and to confirm whether the statistics is correct. This step will be implicitly done in stage 2 of conditional compilation, skip it, if not needed.

  .. code-block:: sh

    python3.8 ../../src/common/conditional_compilation/scripts/ccheader.py --stat ${csv_files} --out conditional_compilation_gen.h

    The conditional_compilation_gen.h looks like this:

  .. code-block:: cpp

    #pragma once

    #define tbb_bind_TBBbindSystemTopology 1
    #define tbb_bind_task_arena_initialize 1

    #define ov_opset_opset1_Parameter 1
    #define ov_opset_opset1_Constant 1
    #define ov_opset_opset1_Convolution 1
    #define ov_opset_opset1_Add 1
    #define ov_opset_opset1_Relu 1
    #define ov_opset_opset1_GroupConvolution 1
    #define ov_opset_opset3_ShapeOf 1
    #define ov_opset_opset1_Squeeze 1
    #define ov_opset_opset4_Range 1
    #define ov_opset_opset1_ReduceMean 1
    #define ov_opset_opset1_Softmax 1
    #define ov_opset_opset1_Result 1

    #define ov_op_v0_Parameter_visit_attributes 1
    #define ov_op_v0_Parameter_validate_and_infer_types 1
    #define ov_op_v0_Parameter_clone_with_new_inputs 1
    #define ov_op_v0_Constant_visit_attributes 1
    #define ov_op_v0_Constant_clone_with_new_inputs 1
    #define ov_op_v1_Convolution_visit_attributes 1
    #define ov_op_v1_Convolution_validate_and_infer_types 1
    #define ov_op_v1_Convolution_clone_with_new_inputs 1
    #define ov_op_v0_util_BinaryElementwiseArithmetic_visit_attributes 1
    #define ov_op_v1_Add_visit_attributes 1
    #define ov_op_v0_util_BinaryElementwiseArithmetic_validate_and_infer_types 1
    #define ov_op_v1_Add_clone_with_new_inputs 1
    #define ov_op_v0_Relu_visit_attributes 1
    #define ov_op_util_UnaryElementwiseArithmetic_validate_and_infer_types 1
    #define ov_op_v0_Relu_clone_with_new_inputs 1
    #define ov_op_v1_GroupConvolution_visit_attributes 1
    #define ov_op_v1_GroupConvolution_validate_and_infer_types 1
    #define ov_op_v1_GroupConvolution_clone_with_new_inputs 1
    #define ov_op_v3_ShapeOf_visit_attributes 1
    #define ov_op_v3_ShapeOf_validate_and_infer_types 1
    #define ov_op_v3_ShapeOf_clone_with_new_inputs 1
    #define ov_op_v0_Squeeze_visit_attributes 1
    ...


Stage 2: build resulting OpenVINO package
++++++++++++++++++++++++++++++++++++++++++

Based on the statistics information, re-build OpenVINO to generate the optimal binary size of OpenVINO binaries

.. code-block:: sh

    cmake -DSELECTIVE_BUILD=ON -DSELECTIVE_BUILD_STAT=${ABSOLUTE_PATH_TO_STATISTICS_FILES}/*.csv -DENABLE_PROFILING_ITT=OFF ..
    cmake --build .

.. tip::

    The recommended scenario for conditional complication is static build of OpenVINO. In this case you can add ``-DBUILD_SHARED_LIBS=OFF`` to enable static build to get optimal binary size benefit.


Conditional Compilation for Different Instruction Set Architectures(ISAs)
#########################################################################

The steps are almost same with building for different models, except for collecting different statistics data on different ``ISAs``.
Run the target application under the ITT collector for code usage analysis on each ``ISAs``:

.. code-block:: sh

    python thirdparty/itt_collector/runtool/sea_runtool.py --bindir ${OPENVINO_LIBRARY_DIR} -o ${MY_MODEL_RESULT} ! ./benchmark_app -niter 1 -nireq 1 -m ${MY_MODEL}.xml

Put all CSV files together for ``stage 2`` to generate resulting OpenVINO binaries:

.. code-block:: sh

    cmake -DSELECTIVE_BUILD=ON -DSELECTIVE_BUILD_STAT=${ABSOLUTE_PATH_TO_STATISTICS_FILES}/*.csv -DENABLE_PROFILING_ITT=OFF ..
    cmake --build .


Device-agnostic Conditional Compilation (POC)
#############################################
In some cases, adopting conditional compilation is necessary to support multiple different ``SKUs`` (Stock Keeping Unit: is usually a string of numbers and alphabets used by the manufacturer to identify their product), but there may be limitations in collecting statistics information for every target hardware. To achieve this, conditional compilation must be capable of running a model on an accelerator with all previous SKUs.

Conditional compilation requires the initial collection of statistical information to exclude unused code regions, such as ops and kernels. To do this, all included ops and kernels must be executed at least once. For multiple SKUs, it is necessary for all ops and kernels that will be used by any of the SKUs to be encountered at least once in the profiling data. If the profiling is done on a CPU platform, it is impossible without using an emulator.

A simple method is to leverage `SDE <https://www.intel.com/content/www/us/en/developer/articles/license/pre-release-license-agreement-for-software-development-emulator.html>`__ to emulate different CPU's SKU to generate multiple statistics CSV files for different SKUs, for example:

.. code-block:: sh

    for cpu in spr adl tgl icl skl; do
        python ../thirdparty/itt_collector/runtool/sea_runtool.py --bindir ${OPENVINO_LIBRARY_DIR} -o ${MY_MODEL_RESULT} ! sde -$cpu -- ./benchmark_app -niter 1 -nireq 1 -m ${MY_MODEL}.xml
    done

Considering that JIT kernels can be affected by L1/L2/L3 cache size and the number of CPU cores, there also is a simple method to emulate L2/L3 cache size and CPU core's number.

- L2/L3 cache emulation

  Hack the function of get cache size ``unsigned int dnnl::impl::cpu::platform::get_per_core_cache_size(int level)`` to make it return emulated cache size in analyzed stage, the simplest way is to leverage environment variable to pass the emulated cache size, for example:

  .. code-block:: cpp

     #if defined(SELECTIVE_BUILD_ANALYZER)
         if (level == 2) {
             const char* L2_cache_size = std::getenv("OV_CC_L2_CACHE_SIZE");
             if (L2_cache_size) {
                 int size = std::atoi(L2_cache_size);
                 if (size > 0) {
                     return size;
                 }
             }
         } else if (level == 3) {
             const char* L3_cache_size = std::getenv("OV_CC_L3_CACHE_SIZE");
             if (L3_cache_size) {
                 int size = std::atoi(L3_cache_size);
                 if (size > 0) {
                     return size;
                 }
             }
         } else if (level == 1) {
             const char* L1_cache_size = std::getenv("OV_CC_L1_CACHE_SIZE");
             if (L1_cache_size) {
                 int size = std::atoi(L1_cache_size);
                 if (size > 0) {
                     return size;
                 }
             }
         }
     #endif

- CPU core number emulation

  Leverage ``numactl`` tool to control core number.

  .. code-block:: sh

    python thirdparty/itt_collector/runtool/sea_runtool.py --bindir ${OPENVINO_LIBRARY_DIR} -o ${MY_MODEL_RESULT} ! numactl -C 0-$core_num ./benchmark_app -niter 1 -nireq 1 -m ${MY_MODEL}.xml


Once the SKUs are decided, you can collect CPU information(CPUID, L1/L2/L3 cache size, core number) and then profile each pair of (CPUID, L1/L2/L3 cache size, core number) to get profiling CSV files, then apply all CSV files to generate final conditional compilation package.

Example of generation a conditional compilation package on a single system:

.. code-block:: sh

    export OV_CC_L1_CACHE_SIZE=<L1 cache size>
    export OV_CC_L2_CACHE_SIZE=<L2 cache size>
    export OV_CC_L3_CACHE_SIZE=<L3 cache size>
    python thirdparty/itt_collector/runtool/sea_runtool.py --bindir ${OPENVINO_LIBRARY_DIR} -o ${MY_MODEL_RESULT} ! sde -spr -- numactl -C 0-$core_num ./benchmark_app -niter 1 -nireq 1 -m ${MY_MODEL}.xml

Perform the above steps for each SKUs information (CPUID, L1/L2/L3 cache size, core number) to collect all generated statistics CSV files together, and provide them to build resulting OpenVINO package.

.. code-block:: sh

    cmake -DSELECTIVE_BUILD=ON -DSELECTIVE_BUILD_STAT=${ABSOLUTE_PATH_TO_STATISTICS_FILES}/*.csv -DENABLE_PROFILING_ITT=OFF ..
    cmake --build .


How to Enable Conditional Compilation on Windows
################################################

Find detailed information in the Building OpenVINO static libraries `document <https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/static_libaries.md>`__ .


Stage 1: Selective build analyzed stage
++++++++++++++++++++++++++++++++++++++++

Build OpenVINO with conditional compilation enabled:

.. code-block:: sh

    call C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvar64.bat
    set OPENVINO_HOME=D:\work_path\openvino
    cd %OPENVINO_HOME%
    md build_cc
    cd build_cc
    cmake -G Ninja -Wno-dev -DCMAKE_BUILD_TYPE=Debug -DENABLE_CPPLINT=OFF -DCMAKE_VERBOSE_MAKEFILE=ON -DCMAKE_COMPILE_WARNING_AS_ERROR=OFF -DENABLE_FASTER_BUILD=ON -DENABLE_SANITIZER=OFF -DTHREADING=TBB -DBUILD_SHARED_LIBS=OFF -DENABLE_PROFILING_ITT=ON -DSELECTIVE_BUILD=COLLECT -DENABLE_INTEL_GPU=OFF -DENABLE_MULTI=OFF -DENABLE_AUTO=OFF -DENABLE_AUTO_BATCH=OFF -DENABLE_HETERO=OFF -DENABLE_TEMPLATE=OFF -DENABLE_OV_ONNX_FRONTEND=OFF -DENABLE_OV_PADDLE_FRONTEND=OFF -DENABLE_OV_PYTORCH_FRONTEND=OFF -DENABLE_OV_JAX_FRONTEND=OFF -DENABLE_OV_TF_FRONTEND=OFF -DCMAKE_INSTALL_PREFIX=install ..

    cmake --build . --config Debug


Collect statistics data

.. code-block:: sh

    cd %OPENVINO_HOME%\build_cc
    cmake --build . --config Debug --target sea_itt_lib
    cd %OPENVINO_HOME%
    set PATH=%PATH%;%OPENVINO_HOME%\\temp\tbb\bin
    mkdir cc_data
    cd %OPENVINO_HOME%\cc_data
    python3 ..\thirdparty\itt_collector\runtool\sea_runtool.py --bindir ..\bin\intel64\Debug -o %OPENVINO_HOME%\cc_data\data ! ..\bin\intel64\Debug\benchmark_app.exe -niter 1 -nireq 1 -m <your_model.xml>

.. note::

    This stage is for profiling data. The choice of whether to build with Debug or Release depends on your specific requirements.


Stage 2: build resulting OpenVINO package
+++++++++++++++++++++++++++++++++++++++++

Generate final optimal binaries size of OpenVINO package

.. code-block:: sh

    cd %OPENVINO_HOME%
    md build
    cd build

    cmake -G "Visual Studio 16 2019" -A x64 -DENABLE_CPPLINT=OFF -DCMAKE_VERBOSE_MAKEFILE=ON -DCMAKE_COMPILE_WARNING_AS_ERROR=OFF -DCMAKE_BUILD_TYPE=Release -DENABLE_FASTER_BUILD=ON -DENABLE_PROFILING_ITT=OFF -DSELECTIVE_BUILD=ON -DENABLE_INTEL_GPU=OFF -DENABLE_MULTI=OFF -DENABLE_AUTO=OFF -DENABLE_AUTO_BATCH=OFF -DENABLE_HETERO=OFF -DENABLE_TEMPLATE=OFF -DENABLE_OV_ONNX_FRONTEND=OFF -DENABLE_OV_PADDLE_FRONTEND=OFF -DENABLE_OV_PYTORCH_FRONTEND=OFF -DENABLE_OV_JAX_FRONTEND=OFF -DENABLE_OV_TF_FRONTEND=OFF -DSELECTIVE_BUILD_STAT=%OPENVINO_HOME%\cc_data\*.csv -DBUILD_SHARED_LIBS=OFF -DENABLE_LTO=ON -DENABLE_ONEDNN_FOR_GPU=OFF -DENABLE_OV_TF_LITE_FRONTEND=OFF -DENABLE_PROFILING_FIRST_INFERENCE=OFF ..

    cmake --build . --config Release


.. tip::

    ``-DSELECTIVE_BUILD=ON`` and ``-DSELECTIVE_BUILD_STAT=%OPENVINO_HOME%\cc_data\*.csv`` are required, and ``-DBUILD_SHARED_LIBS=OFF`` is recommended.

