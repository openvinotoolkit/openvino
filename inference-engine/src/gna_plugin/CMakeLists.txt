# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(TARGET_NAME "GNAPlugin")


file(GLOB_RECURSE SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

file(GLOB_RECURSE HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

find_package(libGNA)

set(TARGET_INCLUDE_DIRS
        ${CMAKE_SOURCE_DIR}/src/inference_engine
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR})

add_definitions(-D_NO_MKL_)

ie_add_plugin(NAME ${TARGET_NAME}
              DEVICE_NAME "GNA"
              SOURCES ${SOURCES} ${HEADERS})

if (LINUX)
    find_package(Threads)
endif()

#saving rpath to GNA shared library be used by CI
log_rpath_from_dir(GNA ${libGNA_LIBRARIES_BASE_PATH})

target_link_libraries(${TARGET_NAME} PRIVATE inference_engine ${INTEL_ITT_LIBS} ${CMAKE_THREAD_LIBS_INIT} libGNA)
target_include_directories(${TARGET_NAME} PUBLIC ${TARGET_INCLUDE_DIRS})

set(TEST_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/gna_plugin.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/gna_pass_manager.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/quantization/quantization.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/dnn.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/gna_device.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/pwl_design.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/floatmath.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/dnn_memory.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/util.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/gna_model_serial.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/gna_plugin_query_api.cpp")

add_library(${TARGET_NAME}_test_static STATIC ${TEST_SOURCES} ${HEADERS})
target_compile_definitions(${TARGET_NAME}_test_static
        PUBLIC -DINTEGER_LOW_P
               -DUSE_STATIC_IE)
target_link_libraries(${TARGET_NAME}_test_static PUBLIC libGNA::API)
target_include_directories(${TARGET_NAME}_test_static PUBLIC ${TARGET_INCLUDE_DIRS})

set_target_properties(${TARGET_NAME}_test_static PROPERTIES COMPILE_PDB_NAME ${TARGET_NAME}_test_static)
