# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# RPATH is always enabled for unit tests
set(CMAKE_SKIP_RPATH OFF)

add_subdirectory(inference_engine)

if (ENABLE_MKL_DNN)
    add_subdirectory(cpu)
endif ()

if (ENABLE_GNA)
    add_subdirectory(gna)
endif ()

if (ENABLE_MYRIAD)
    add_subdirectory(vpu)
endif ()

if(NGRAPH_ONNX_IMPORT_ENABLE)
    add_subdirectory(frontends/onnx_import)
endif()
