# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME gpuFuncTests)

addIeTargetTest(
        NAME
            ${TARGET_NAME}
        ROOT
            ${CMAKE_CURRENT_SOURCE_DIR}
        INCLUDES
            ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDENCIES
            clDNNPlugin
            AutoPlugin
        LINK_LIBRARIES
            funcSharedTests
            OpenCL
        ADD_CPPLINT
        LABELS
            GPU
)

# CVS-55376
set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/low_precision_transformations/layer_transformation.cpp"
    PROPERTIES INCLUDE_DIRECTORIES
    $<TARGET_PROPERTY:inference_engine_legacy,INTERFACE_INCLUDE_DIRECTORIES>)

# try to find VA libraries
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_search_module(LIBVA QUIET libva)
endif()

# TODO: pkg_search_module finds libva not in sysroot
if(ANDROID)
    set(LIBVA_FOUND OFF CACHE BOOL "" FORCE)
endif()

if(LIBVA_FOUND)
    target_compile_definitions(${TARGET_NAME} PRIVATE ENABLE_LIBVA)
    target_include_directories(${TARGET_NAME} PRIVATE ${LIBVA_INCLUDE_DIRS})
    target_link_libraries(${TARGET_NAME} PRIVATE ${LIBVA_LINK_LIBRARIES})
endif()
