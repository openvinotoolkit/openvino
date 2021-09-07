# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(SUGGEST_OVERRIDE_SUPPORTED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-suggest-override")
endif()

if(ENABLE_LTO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()

add_subdirectory(ittapi EXCLUDE_FROM_ALL)
add_subdirectory(itt_collector EXCLUDE_FROM_ALL)
add_subdirectory(zlib EXCLUDE_FROM_ALL)
add_subdirectory(cnpy EXCLUDE_FROM_ALL)
add_subdirectory(ocl EXCLUDE_FROM_ALL)

add_subdirectory(xbyak EXCLUDE_FROM_ALL)
openvino_developer_export_targets(COMPONENT openvino_common TARGETS xbyak)

#
# Pugixml
#

if(ENABLE_SYSTEM_PUGIXML)
    find_package(PugiXML REQUIRED)
    set_property(TARGET pugixml PROPERTY IMPORTED_GLOBAL TRUE)
    add_library(pugixml::static ALIAS pugixml)
else()
    function(ie_build_pugixml)
        set(BUILD_SHARED_LIBS OFF)
        add_subdirectory(pugixml EXCLUDE_FROM_ALL)
        set_property(TARGET pugixml-static PROPERTY EXPORT_NAME pugixml)
        openvino_developer_export_targets(COMPONENT openvino_common TARGETS pugixml::static)
    endfunction()

    ie_build_pugixml()
endif()

#
# Fluid, G-API, OpenCV HAL
#

add_library(ocv_hal INTERFACE)
target_include_directories(ocv_hal INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/ocv")

add_subdirectory(ade EXCLUDE_FROM_ALL)
add_subdirectory(fluid/modules/gapi EXCLUDE_FROM_ALL)
set_target_properties(ade fluid PROPERTIES FOLDER thirdparty)
openvino_developer_export_targets(COMPONENT openvino_common TARGETS ade fluid)

#
# Gflags
#

add_subdirectory(gflags EXCLUDE_FROM_ALL)
openvino_developer_export_targets(COMPONENT openvino_common TARGETS gflags)

#
# Google Tests framework
#

if(NGRAPH_UNIT_TEST_ENABLE OR ENABLE_TESTS)
    add_subdirectory(gtest EXCLUDE_FROM_ALL)

    openvino_developer_export_targets(COMPONENT inference_engine_tests
                                      TARGETS gmock gmock_main gtest gtest_main)
endif()

#
# Protobuf
#

if(NGRAPH_PDPD_FRONTEND_ENABLE OR NGRAPH_ONNX_FRONTEND_ENABLE)
    if(NGRAPH_USE_SYSTEM_PROTOBUF)
        set(Protobuf_USE_STATIC_LIBS ON)
        if(VERBOSE_BUILD)
            set(Protobuf_DEBUG ON)
        endif()
        find_package(Protobuf 3.9.0 REQUIRED)
        set(Protobuf_LITE_LIBRARIES protobuf::libprotobuf-lite)
        set(SYSTEM_PROTOC protobuf::protoc)
        set(PROTOC_EXECUTABLE ${SYSTEM_PROTOC})

        foreach(target ${SYSTEM_PROTOC} ${Protobuf_LITE_LIBRARIES})
            set_property(TARGET ${target} PROPERTY IMPORTED_GLOBAL TRUE)
        endforeach()
    else()
        add_subdirectory(protobuf)
    endif()

    # forward variables used in the other places
    set(SYSTEM_PROTOC ${SYSTEM_PROTOC} PARENT_SCOPE)
    set(PROTOC_EXECUTABLE ${PROTOC_EXECUTABLE} PARENT_SCOPE)
    set(Protobuf_LITE_LIBRARIES ${Protobuf_LITE_LIBRARIES} PARENT_SCOPE)
    set(Protobuf_INCLUDE_DIRS ${Protobuf_INCLUDE_DIRS} PARENT_SCOPE)

    # set public / interface compile options
    foreach(target IN LISTS Protobuf_LITE_LIBRARIES)
        set(link_type PUBLIC)
        if(NGRAPH_USE_SYSTEM_PROTOBUF)
            set(link_type INTERFACE)
        endif()
        if(CMAKE_COMPILER_IS_GNUCXX OR OV_COMPILER_IS_CLANG)
            target_compile_options(${target} ${link_type} -Wno-undef)
        endif()
    endforeach()
endif()

#
# ONNX
#

if(NGRAPH_ONNX_FRONTEND_ENABLE)
    add_subdirectory(onnx)
endif()

#
# oneDNN for GPU plugin
#
if (ANDROID OR (CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0))
    # oneDNN doesn't support old compilers and android builds for now, so we'll build GPU plugin without oneDNN
    set(ENABLE_ONEDNN_FOR_GPU OFF CACHE BOOL "")
else()
    set(ENABLE_ONEDNN_FOR_GPU ON CACHE BOOL "")
endif()

if(ENABLE_ONEDNN_FOR_GPU)
    function(build_onednn_gpu)
        include(ExternalProject)
        set(ONEDNN_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/onednn_gpu_build/")
        set(ONEDNN_INSTALL_DIR "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
        set(ONEDNN_LIBRARY "${ONEDNN_BUILD_DIR}/src/${CMAKE_STATIC_LIBRARY_PREFIX}onednn_gpu${CMAKE_STATIC_LIBRARY_SUFFIX}")
        set(ONEDNN_OUTPUT_LIBRARY "${ONEDNN_INSTALL_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}onednn_gpu${CMAKE_STATIC_LIBRARY_SUFFIX}")
        ExternalProject_Add(onednn_gpu_build
            SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/onednn_gpu"
            BINARY_DIR "${ONEDNN_BUILD_DIR}"
            INSTALL_DIR "${ONEDNN_INSTALL_DIR}"
            PREFIX onednn_gpu_build
            EXCLUDE_FROM_ALL ON
            CMAKE_ARGS "-DDNNL_CPU_RUNTIME=NONE"
                       "-DDNNL_GPU_RUNTIME=OCL"
                       "-DDNNL_LIBRARY_NAME=onednn_gpu"
                       "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
                       "-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
                       "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
                       "-DDNNL_ENABLE_CONCURRENT_EXEC=ON"
                       "-DDNNL_ENABLE_PRIMITIVE_CACHE=ON"
                       "-DDNNL_BUILD_TESTS=OFF"
                       "-DDNNL_BUILD_EXAMPLES=OFF"
                       "-DDNNL_BLAS_VENDOR=NONE"
                       "-DDNNL_LIBRARY_TYPE=STATIC"
                       "-DOpenCL_LIBRARY=${OpenCL_LIBRARY}"
                       "-DOpenCL_INCLUDE_DIR=${OpenCL_INCLUDE_DIR}"
                       "-DOPENCL_VERSION_2_2=${OPENCL_VERSION_2_2}"
            BUILD_COMMAND ${CMAKE_COMMAND} --build . --config ${CMAKE_BUILD_TYPE} --target onednn_gpu --parallel
            INSTALL_COMMAND ${CMAKE_COMMAND} -E copy ${ONEDNN_LIBRARY} ${ONEDNN_OUTPUT_LIBRARY}
            COMMAND ${CMAKE_COMMAND} -E echo "OneDNN $<CONFIG> build for GPU complete"
            BUILD_BYPRODUCTS ${ONEDNN_OUTPUT_LIBRARY}
        )
    endfunction()

    build_onednn_gpu()
endif()

#
# Install
#

ie_cpack_add_component(cpp_samples_deps)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/gflags
        DESTINATION "samples/cpp/thirdparty"
        COMPONENT cpp_samples_deps
        USE_SOURCE_PERMISSIONS)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/zlib
        DESTINATION "samples/cpp/thirdparty"
        COMPONENT cpp_samples_deps
        USE_SOURCE_PERMISSIONS)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cnpy
        DESTINATION "samples/cpp/thirdparty"
        COMPONENT cpp_samples_deps
        USE_SOURCE_PERMISSIONS)
