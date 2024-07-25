# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if (NOT ENABLE_INTEL_NPU)
    return()
endif()

#
# Build properties
#

set(NPU_DEVICE_NAME "NPU")
string(TOLOWER "${NPU_DEVICE_NAME}" NPU_PLUGIN_COMPONENT)
set(NPU_INTERNAL_COMPONENT "${NPU_PLUGIN_COMPONENT}_internal")

set(NPU_PLUGIN_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

include(cmake/features.cmake)

if(MSVC AND BUILD_COMPILER_FOR_DRIVER)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        # /Zi flag is used for making debug info added to pdb files instead of the library itself
        # /DEBUG flag is used for adding PDB file to exe files, shared libraries, static libraries
        # /OPT:REF and /OPT:ICF are used for decreasing the size of PDB files
        # /INCREMENTAL:NO is used for disable incremental linking
        # /FS flag forces pdb file write to MSPDBSRV.exe
        # /Zf flag reduces the RPC call to MSPDBSRV.exe
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Zi /FS /Zf /ZH:SHA_256 /guard:cf /Qspectre")
        set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /Zi /FS /Zf /ZH:SHA_256 /guard:cf /Qspectre")
        set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /DEBUG /OPT:REF /OPT:ICF /INCREMENTAL:NO /CETCOMPAT /guard:cf")
        set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /DEBUG /OPT:REF /OPT:ICF /INCREMENTAL:NO /CETCOMPAT /guard:cf")
        set(CMAKE_MODULE_LINKER_FLAGS_RELEASE "${CMAKE_MODULE_LINKER_FLAGS_RELEASE} /DEBUG /OPT:REF /OPT:ICF /INCREMENTAL:NO /CETCOMPAT /guard:cf")
    endif()
endif()

set(CMAKE_CXX_STANDARD 17)

if(ENABLE_NPU_DEBUG_CAPS)
    add_compile_definitions(NPU_PLUGIN_DEVELOPER_BUILD)
endif()

add_subdirectory(thirdparty EXCLUDE_FROM_ALL)

add_subdirectory(src)

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()

add_subdirectory(tools)

ov_cpack_add_component(${NPU_INTERNAL_COMPONENT} HIDDEN)
