# Copyright (C) 2018-2020 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ClDnnFunctionalTests)

file(GLOB CLDNN_TEST_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

list(APPEND TEST_SRC ${CLDNN_TEST_SOURCES})

list(APPEND CLDNN_LIBS
        IESharedTests
        ${CLDNN__IOCL_ICD_LIBPATH})

# try to find VA libraries
include(FindPkgConfig)
pkg_search_module(LIBVA QUIET libva)

# TODO: pkg_search_module finds libva not in sysroot
if(ANDROID)
    set(LIBVA_FOUND OFF CACHE BOOL "" FORCE)
endif()

if(LIBVA_FOUND)
    list(APPEND CLDNN_LIBS ${LIBVA_LINK_LIBRARIES})
endif()

list(APPEND DEPENDENCIES
        clDNNPlugin)

if (ENABLE_MKL_DNN)
    list(APPEND DEPENDENCIES
            MKLDNNPlugin
            HeteroPlugin)
endif()

# add OpenCL dependency end

source_group("src" FILES ${TEST_SRC})
source_group("include" FILES ${TEST_INCLUDE})

add_executable(${TARGET_NAME}
        ${TEST_SRC}
        ${TEST_INCLUDE})

target_compile_definitions(${TARGET_NAME}
        PUBLIC ${ARGV}
        DATA_PATH=\"${DATA_PATH}\"
        MODELS_PATH=\"${MODELS_PATH}\")

if(LIBVA_FOUND)
    target_compile_definitions(${TARGET_NAME} PRIVATE ENABLE_LIBVA)
    target_include_directories(${TARGET_NAME} PRIVATE ${LIBVA_INCLUDE_DIRS})
endif()

target_include_directories(${TARGET_NAME} PRIVATE ${CLDNN__IOCL_ICD_INCDIRS})
target_link_libraries(${TARGET_NAME} PRIVATE ${CLDNN_LIBS})

add_dependencies(${TARGET_NAME} ${DEPENDENCIES})

add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})
set_property(TEST ${TARGET_NAME} PROPERTY LABELS GPU)
