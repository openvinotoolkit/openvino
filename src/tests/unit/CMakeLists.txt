# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# RPATH is always enabled for unit tests
set(CMAKE_SKIP_RPATH OFF)

# because unit tests use plugins object files compiled with LTO
if(CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 9.0)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})
endif()

add_subdirectory(inference_engine)

if (ENABLE_INTEL_CPU)
    add_subdirectory(cpu)
endif ()

if (ENABLE_INTEL_GNA)
    add_subdirectory(gna)
endif ()

if (ENABLE_INTEL_MYRIAD)
    add_subdirectory(vpu)
endif ()

if(ENABLE_OV_ONNX_FRONTEND)
    add_subdirectory(frontends/onnx_import)
endif()

if (ENABLE_AUTO OR ENABLE_MULTI)
    add_subdirectory(auto)
endif()
