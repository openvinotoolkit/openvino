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
set(ENABLE_ONEDNN_FOR_GPU ON CACHE BOOL "")
if(ENABLE_ONEDNN_FOR_GPU)
    function(build_onednn_gpu)
        set(DNNL_ENABLE_CONCURRENT_EXEC ON CACHE BOOL "" FORCE)
        set(DNNL_ENABLE_PRIMITIVE_CACHE ON CACHE BOOL "" FORCE)
        set(DNNL_USE_RT_OBJECTS_IN_PRIMITIVE_CACHE OFF CACHE BOOL "" FORCE)
        set(DNNL_ENABLE_MAX_CPU_ISA OFF CACHE BOOL "" FORCE)
        set(DNNL_LIBRARY_TYPE STATIC CACHE BOOL "" FORCE)
        set(DNNL_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(DNNL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(DNNL_GPU_RUNTIME "OCL" CACHE STRING "" FORCE)
        set(DNNL_CPU_RUNTIME "NONE" CACHE STRING "" FORCE)
        set(DNNL_BLAS_VENDOR "NONE" CACHE STRING "" FORCE)
        set(SDL_cmake_included ON)  ## to skip internal SDL flags. SDL flags are already set on IE level
        if (ANDROID OR ((CMAKE_CXX_COMPILER_ID STREQUAL "MSVC") AND NOT (THREADING STREQUAL "OMP")))
            set(OpenMP_cmake_included ON) ## to skip "omp simd" inside a code. Lead to some crashes inside NDK LLVM..
        endif()

        # Workaround for TBB headers imported from IE
        if(NOT WIN32)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-undef")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-undef")
        endif()

        set(DNNL_LIBRARY_NAME onednn_gpu CACHE STRING "" FORCE)
        add_subdirectory(onednn_gpu "${DNNL_LIBRARY_NAME}" EXCLUDE_FROM_ALL)
        set_property(GLOBAL PROPERTY DNNL_LIB_DEPS "")
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
