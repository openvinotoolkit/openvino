# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "MKLDNNPlugin")

if (UNIX AND NOT APPLE)
    set(OS_SPECIFIC_SRC ${CMAKE_CURRENT_SOURCE_DIR}/mkldnn/os/lin/*.cpp)
endif()

file(GLOB SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/mkldnn/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/utils/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/nodes/*.cpp
        ${OS_SPECIFIC_SRC}
)

file(GLOB HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/mkldnn/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/mkldnn/*.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/utils/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/nodes/*.h
)

addVersionDefines(mkldnn_plugin.cpp CI_BUILD_NUMBER MKL_VERSION)

add_definitions(-DIMPLEMENT_INFERENCE_ENGINE_PLUGIN)

include_directories(
        ${IE_MAIN_SOURCE_DIR}/include
        ${IE_MAIN_SOURCE_DIR}/src/inference_engine
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/mkldnn
        ${IE_MAIN_SOURCE_DIR}/thirdparty/mkl-dnn/src/common
        ${IE_MAIN_SOURCE_DIR}/thirdparty/mkl-dnn/src/cpu
        ${IE_MAIN_SOURCE_DIR}/thirdparty/mkl-dnn/include
        ${CMAKE_BINARY_DIR}/include/
)

if (GEMM STREQUAL "MKL")
    log_rpath_from_dir(MKL "${MKL}/lib")
endif()

add_library(${TARGET_NAME} SHARED ${SOURCES} ${HEADERS})
set_ie_threading_interface_for(${TARGET_NAME})

if (THREADING STREQUAL "TBB")
    set(MKLDNN_THR MKLDNN_THR_TBB)
elseif (THREADING STREQUAL "OMP")
    set(MKLDNN_THR MKLDNN_THR_OMP)
else()
    set(MKLDNN_THR MKLDNN_THR_SEQ)
endif()

target_compile_definitions(${TARGET_NAME} PUBLIC -DMKLDNN_THR=${MKLDNN_THR})
target_link_libraries(${TARGET_NAME} PRIVATE inference_engine ${INTEL_ITT_LIBS} mkldnn)

set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_PDB_NAME ${TARGET_NAME})

add_library(test_${TARGET_NAME} STATIC ${SOURCES} ${HEADERS})
set_ie_threading_interface_for(test_${TARGET_NAME})

target_compile_definitions(test_${TARGET_NAME} PUBLIC -DMKLDNN_THR=${MKLDNN_THR})
target_link_libraries(test_${TARGET_NAME} PRIVATE inference_engine_s mkldnn)

set_target_properties(test_${TARGET_NAME} PROPERTIES COMPILE_PDB_NAME test_${TARGET_NAME})

add_cpplint_target(${TARGET_NAME}_cpplint FOR_TARGETS ${TARGET_NAME})
