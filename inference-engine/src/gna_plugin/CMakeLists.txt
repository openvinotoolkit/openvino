# Copyright (C) 2018 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "GNAPlugin")

file(GLOB_RECURSE SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        )

file(GLOB_RECURSE HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
        )

add_definitions(-DIMPLEMENT_INFERENCE_ENGINE_PLUGIN)

find_package(libGNA)
include_directories(${libGNA_INCLUDE_DIRS})

include_directories(
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src/inference_engine
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${libGNA_INCLUDE_DIRS}
)

add_definitions(-D_NO_MKL_)
add_library(${TARGET_NAME} SHARED ${SOURCES} ${HEADERS})

if (LINUX)
    find_package(Threads)
endif ()

set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_PDB_NAME ${TARGET_NAME})

#saving rpath to GNA shared library be used by CI
log_rpath_remove_top(GNA FALSE "/gna${libGNA_LIBRARY}" TRUE)

target_link_libraries(${TARGET_NAME} inference_engine ${INTEL_ITT_LIBS} ${libGNA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})


set(TEST_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/gna_plugin.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/gna_plugin_passes.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/quantization/quantization.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/dnn.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/gna_device.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/pwl_design.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/floatmath.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/dnn_memory.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/util.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/gna_model_serial.cpp")

add_library(${TARGET_NAME}_test_static STATIC ${TEST_SOURCES} ${HEADERS})
target_compile_definitions(${TARGET_NAME}_test_static
        PUBLIC -DINTEGER_LOW_P
               -DUSE_STATIC_IE)

set_target_properties(${TARGET_NAME}_test_static PROPERTIES COMPILE_PDB_NAME ${TARGET_NAME}_test_static)
