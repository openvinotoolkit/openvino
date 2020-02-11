# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set (TARGET_NAME "inference_engine")

if(ENABLE_LTO)
    ie_enable_lto()
endif()

set(builder_files_src ${CMAKE_CURRENT_SOURCE_DIR}/builders/ie_layer_decorator.cpp
                      ${CMAKE_CURRENT_SOURCE_DIR}/builders/ie_layer_builder.cpp
                      ${CMAKE_CURRENT_SOURCE_DIR}/builders/ie_input_layer_layer.cpp
                      ${CMAKE_CURRENT_SOURCE_DIR}/builders/ie_const_layer.cpp
                      ${CMAKE_CURRENT_SOURCE_DIR}/builders/ie_split_layer.cpp
                      ${CMAKE_CURRENT_SOURCE_DIR}/builders/ie_pooling_layer.cpp
                      ${CMAKE_CURRENT_SOURCE_DIR}/builders/ie_network_builder.cpp)

file (GLOB LIBRARY_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/low_precision_transformations/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/low_precision_transformations/common/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/transform/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/transform/transformations/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/transform/transformations/fusion/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/transform/transformations/utils/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ngraph_ops/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cpp_interfaces/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/shape_infer/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/shape_infer/built-in/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/shape_infer/const_infer/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cpp_interfaces/impl/*.cpp
        ${builder_files_src}
      )

if (ENABLE_NGRAPH)
    file (GLOB LIBRARY_SRC ${LIBRARY_SRC}
        ${CMAKE_CURRENT_SOURCE_DIR}/transform/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/transform/transformations/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ngraph_ops/*.cpp)
endif()

file (GLOB LIBRARY_HEADERS
       ${CMAKE_CURRENT_SOURCE_DIR}/*.h
       ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
       ${CMAKE_CURRENT_SOURCE_DIR}/cpp_interfaces/*.hpp
       ${CMAKE_CURRENT_SOURCE_DIR}/shape_infer/*.hpp
       ${CMAKE_CURRENT_SOURCE_DIR}/shape_infer/built-in/*.hpp
       ${CMAKE_CURRENT_SOURCE_DIR}/shape_infer/const_infer/*.hpp
       ${CMAKE_CURRENT_SOURCE_DIR}/cpp_interfaces/base/*.hpp
       ${CMAKE_CURRENT_SOURCE_DIR}/cpp_interfaces/impl/*.hpp
       ${CMAKE_CURRENT_SOURCE_DIR}/cpp_interfaces/interface/*.hpp
       ${CMAKE_CURRENT_SOURCE_DIR}/low_precision_transformations/*.hpp
       ${CMAKE_CURRENT_SOURCE_DIR}/low_precision_transformations/common/*.hpp
      )

if (NOT ENABLE_NGRAPH)
    list(REMOVE_ITEM LIBRARY_SRC "${CMAKE_CURRENT_SOURCE_DIR}/ie_cnn_layer_builder_ngraph.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/cnn_network_ngraph_impl.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ie_ir_parser.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ie_ir_reader.cpp")
    list(REMOVE_ITEM LIBRARY_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/ie_ngraph_utils.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/cnn_network_ngraph_impl.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ie_cnn_layer_builder_ngraph.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/ie_ir_parser.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ie_ir_reader.hpp")
endif()

if( (NOT DEFINED ENABLE_SSE42) OR ENABLE_SSE42)
    file (GLOB LIBRARY_SRC
           ${LIBRARY_SRC}
           ${CMAKE_CURRENT_SOURCE_DIR}/cpu_x86_sse42/*.cpp
          )
    file (GLOB LIBRARY_HEADERS
           ${LIBRARY_HEADERS}
           ${CMAKE_CURRENT_SOURCE_DIR}/cpu_x86_sse42/*.hpp
          )

    file (GLOB SSE_SRC
           ${CMAKE_CURRENT_SOURCE_DIR}/cpu_x86_sse42/*.cpp
          )
    file (GLOB SSE_HEADERS
           ${CMAKE_CURRENT_SOURCE_DIR}/cpu_x86_sse42/*.hpp
          )

    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cpu_x86_sse42)

    if (WIN32)
        if(CMAKE_CXX_COMPILER_ID MATCHES MSVC)
            # No such option for MSVC 2019
        elseif(CMAKE_CXX_COMPILER_ID STREQUAL Intel)
            set_source_files_properties(${SSE_SRC}
                PROPERTIES COMPILE_FLAGS /arch:SSE4.2 /QxSSE4.2 /Qvc14)
        elseif(CMAKE_CXX_COMPILER_ID STREQUAL Clang)
            set_source_files_properties(${SSE_SRC}
                PROPERTIES COMPILE_FLAGS -msse4.2)
        endif()
    else()
        set_source_files_properties(${SSE_SRC}
                PROPERTIES COMPILE_FLAGS -msse4.2)
    endif()
    add_definitions(-DHAVE_SSE=1)
endif()

addVersionDefines(ie_version.cpp CI_BUILD_NUMBER)

set (PUBLIC_HEADERS_DIR "${IE_MAIN_SOURCE_DIR}/include")

file (GLOB PUBLIC_HEADERS
       ${PUBLIC_HEADERS_DIR}/builders/*.hpp
       ${PUBLIC_HEADERS_DIR}/builders/*.h
       ${PUBLIC_HEADERS_DIR}/cpp/*.hpp
       ${PUBLIC_HEADERS_DIR}/cpp/*.h
       ${PUBLIC_HEADERS_DIR}/details/os/*.hpp
       ${PUBLIC_HEADERS_DIR}/details/os/*.h
       ${PUBLIC_HEADERS_DIR}/details/*.hpp
       ${PUBLIC_HEADERS_DIR}/details/*.h
       ${PUBLIC_HEADERS_DIR}/*.hpp
       ${PUBLIC_HEADERS_DIR}/*.h
      )

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj

source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${LIBRARY_HEADERS} ${PUBLIC_HEADERS})

# Plugin API library

add_library(${TARGET_NAME}_plugin_api INTERFACE)

target_include_directories(${TARGET_NAME}_plugin_api INTERFACE
    "${IE_MAIN_SOURCE_DIR}/src/inference_engine"
    $<TARGET_PROPERTY:${TARGET_NAME}_preproc,INTERFACE_INCLUDE_DIRECTORIES>
    ${PUBLIC_HEADERS_DIR})

if(ENABLE_NGRAPH)
    target_compile_definitions(${TARGET_NAME}_plugin_api INTERFACE ENABLE_NGRAPH)
endif()

# Create object library

add_library(${TARGET_NAME}_obj OBJECT
            ${LIBRARY_SRC}
            ${LIBRARY_HEADERS}
            ${PUBLIC_HEADERS})

target_compile_definitions(${TARGET_NAME}_obj PRIVATE IMPLEMENT_INFERENCE_ENGINE_API)

if (ENABLE_NGRAPH)
    target_include_directories(${TARGET_NAME}_obj SYSTEM PRIVATE $<TARGET_PROPERTY:ngraph::ngraph,INTERFACE_INCLUDE_DIRECTORIES>)
endif()
target_include_directories(${TARGET_NAME}_obj SYSTEM PRIVATE "${IE_MAIN_SOURCE_DIR}/thirdparty/pugixml/src")

target_include_directories(${TARGET_NAME}_obj PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}"
                                                      $<TARGET_PROPERTY:inference_engine_preproc,INTERFACE_INCLUDE_DIRECTORIES>
                                                      $<TARGET_PROPERTY:${TARGET_NAME}_plugin_api,INTERFACE_INCLUDE_DIRECTORIES>)

if(ENABLE_PROFILING_ITT AND INTEL_ITT_LIBS)
    target_include_directories(${TARGET_NAME}_obj SYSTEM PRIVATE $<TARGET_PROPERTY:ittnotify,INTERFACE_INCLUDE_DIRECTORIES>)
    target_compile_definitions(${TARGET_NAME}_obj PRIVATE $<TARGET_PROPERTY:ittnotify,INTERFACE_COMPILE_DEFINITIONS>)
endif()

if(ENABLE_MKL_DNN)
    target_include_directories(${TARGET_NAME}_obj SYSTEM PRIVATE "${IE_MAIN_SOURCE_DIR}/thirdparty/mkl-dnn/src/cpu/xbyak")
endif()

set_ie_threading_interface_for(${TARGET_NAME}_obj)

if(WIN32)
    # To disable min/max macro in windows.h
    target_compile_definitions(${TARGET_NAME}_obj PRIVATE -DNOMINMAX)
endif()

if(ENABLE_IR_READER)
    target_compile_definitions(${TARGET_NAME}_obj PRIVATE ENABLE_IR_READER)
endif()

add_cpplint_target(${TARGET_NAME}_cpplint FOR_TARGETS ${TARGET_NAME}_obj EXCLUDE_PATTERNS "../ngraph/*")
add_clang_format_target(${TARGET_NAME}_clang_format FOR_TARGETS ${TARGET_NAME}_obj EXCLUDE_PATTERNS "../ngraph/*")

# Create shared library file from object library

add_library(${TARGET_NAME} SHARED
            $<TARGET_OBJECTS:${TARGET_NAME}_obj>)

set_ie_threading_interface_for(${TARGET_NAME})

target_link_libraries(${TARGET_NAME} PRIVATE Threads::Threads ${INTEL_ITT_LIBS} pugixml ${NGRAPH_LIBRARIES} PUBLIC ${CMAKE_DL_LIBS})

target_include_directories(${TARGET_NAME} INTERFACE ${PUBLIC_HEADERS_DIR})

if(WIN32)
    set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_PDB_NAME ${TARGET_NAME})
endif()


if(ENABLE_UNICODE_PATH_SUPPORT)
    target_compile_definitions(${TARGET_NAME} PUBLIC ENABLE_UNICODE_PATH_SUPPORT)
endif()

ie_register_plugins(MAIN_TARGET ${TARGET_NAME}
                    POSSIBLE_PLUGINS HeteroPlugin clDNNPlugin GNAPlugin MKLDNNPlugin myriadPlugin)

# Create NN Builder

file(GLOB NN_BUILDER_LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/builders/*.cpp)

# disable deprecated warnings for NN Builder

function(nn_builder_disable_warnings)
    disable_deprecated_warnings()
    set_source_files_properties(${NN_BUILDER_LIBRARY_SRC} PROPERTIES COMPILE_FLAGS ${ie_c_cxx_deprecated})
endfunction()

nn_builder_disable_warnings()

list(REMOVE_ITEM NN_BUILDER_LIBRARY_SRC ${builder_files_src})

add_library(${TARGET_NAME}_nn_builder SHARED ${NN_BUILDER_LIBRARY_SRC})

target_compile_definitions(${TARGET_NAME}_nn_builder PRIVATE IMPLEMENT_INFERENCE_ENGINE_API)

target_link_libraries(${TARGET_NAME}_nn_builder PUBLIC ${TARGET_NAME})

target_include_directories(${TARGET_NAME}_nn_builder PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}"
    $<TARGET_PROPERTY:inference_engine_plugin_api,INTERFACE_INCLUDE_DIRECTORIES>)

# Static library used for unit tests which are always built

add_library(${TARGET_NAME}_s STATIC
            $<TARGET_OBJECTS:${TARGET_NAME}_obj>
            ${NN_BUILDER_LIBRARY_SRC})

set_ie_threading_interface_for(${TARGET_NAME}_s)

target_include_directories(${TARGET_NAME}_s PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
    $<TARGET_PROPERTY:inference_engine_plugin_api,INTERFACE_INCLUDE_DIRECTORIES>)

if(WIN32)
    set_target_properties(${TARGET_NAME}_s PROPERTIES COMPILE_PDB_NAME ${TARGET_NAME}_s)
endif()

if(ENABLE_UNICODE_PATH_SUPPORT)
    target_compile_definitions(${TARGET_NAME}_s PUBLIC ENABLE_UNICODE_PATH_SUPPORT)
endif()

target_link_libraries(${TARGET_NAME}_s PRIVATE ${NGRAPH_LIBRARIES}
# NOTE: public linkage with ITT is important here! without it binaries that use this library do not
#       get proper compile flags to support ITT which causes hard-to-detect bugs.
                                       PUBLIC pugixml ${INTEL_ITT_LIBS} ${CMAKE_DL_LIBS})

target_compile_definitions(${TARGET_NAME}_s PUBLIC USE_STATIC_IE)

if(ENABLE_PROFILING_ITT AND INTEL_ITT_LIBS)
    target_include_directories(${TARGET_NAME}_s SYSTEM PRIVATE $<TARGET_PROPERTY:ittnotify,INTERFACE_INCLUDE_DIRECTORIES>)
    target_compile_definitions(${TARGET_NAME}_s PRIVATE $<TARGET_PROPERTY:ittnotify,INTERFACE_COMPILE_DEFINITIONS>)
endif()

# export targets

export(TARGETS ${TARGET_NAME} ${TARGET_NAME}_nn_builder NAMESPACE IE:: APPEND FILE "${CMAKE_BINARY_DIR}/targets.cmake")

configure_file(
    "${IE_MAIN_SOURCE_DIR}/cmake/config.cmake.in"
    "${CMAKE_BINARY_DIR}/InferenceEngineConfig.cmake"
    COPYONLY)

configure_file(
    "${IE_MAIN_SOURCE_DIR}/cmake/share/InferenceEngineConfig-version.cmake.in"
    "${CMAKE_BINARY_DIR}/InferenceEngineConfig-version.cmake"
    COPYONLY)

# developer package

add_library(xbyak INTERFACE)
target_include_directories(xbyak INTERFACE ${IE_MAIN_SOURCE_DIR}/thirdparty/mkl-dnn/src/cpu/xbyak)

ie_developer_export_targets(${TARGET_NAME} ${TARGET_NAME}_plugin_api ${TARGET_NAME}_nn_builder xbyak)

# install

if(THREADING STREQUAL "TBB" OR THREADING STREQUAL "TBB_AUTO")
    set(threading_tbb ON)
endif()

if(ENABLE_NGRAPH)
    list(APPEND core_components ngraph)
endif()

if(threading_tbb)
    ie_cpack_add_component(tbb REQUIRED)
    list(APPEND core_components tbb)

    install(DIRECTORY "${TBB}/include"
            DESTINATION ${IE_CPACK_IE_DIR}/external/tbb
            COMPONENT tbb)
    install(DIRECTORY "${TBB}/lib"
            DESTINATION ${IE_CPACK_IE_DIR}/external/tbb
            COMPONENT tbb)
    if(EXISTS "${TBB}/bin")
      install(DIRECTORY "${TBB}/bin"
              DESTINATION ${IE_CPACK_IE_DIR}/external/tbb
              COMPONENT tbb)
    endif()
    install(FILES "${TBB}/LICENSE"
            DESTINATION ${IE_CPACK_IE_DIR}/external/tbb
            COMPONENT tbb)
endif()

ie_cpack_add_component(core REQUIRED DEPENDS ${core_components})

install(DIRECTORY "${IE_MAIN_SOURCE_DIR}/include" DESTINATION ${IE_CPACK_IE_DIR}
        COMPONENT core)
install(TARGETS ${TARGET_NAME} ${TARGET_NAME}_nn_builder
        RUNTIME DESTINATION ${IE_CPACK_LIBRARY_PATH}
        ARCHIVE DESTINATION ${IE_CPACK_LIBRARY_PATH}
        LIBRARY DESTINATION ${IE_CPACK_LIBRARY_PATH}
        COMPONENT core)
install(FILES "${OpenVINO_BINARY_DIR}/share/ie_parallel.cmake"
              "${OpenVINO_BINARY_DIR}/share/ie_rh_decoder.cmake"
              "${OpenVINO_BINARY_DIR}/share/InferenceEngineConfig.cmake"
              "${OpenVINO_BINARY_DIR}/share/InferenceEngineConfig-version.cmake"
        DESTINATION ${IE_CPACK_IE_DIR}/share
        COMPONENT core)
install(FILES $<TARGET_FILE_DIR:${TARGET_NAME}>/plugins.xml
        DESTINATION ${IE_CPACK_LIBRARY_PATH}
        COMPONENT core)
