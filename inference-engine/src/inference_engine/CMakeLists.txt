# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set (TARGET_NAME "inference_engine")

if (WIN32)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNOMINMAX")
endif()

file (GLOB LIBRARY_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/transform/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/transform/transformations/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ngraph_ops/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/builders/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cpp_interfaces/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/shape_infer/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/shape_infer/built-in/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/shape_infer/const_infer/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/hetero/*.cpp
      )

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
       ${CMAKE_CURRENT_SOURCE_DIR}/hetero/*.hpp
      )

if( (NOT DEFINED ENABLE_SSE42) OR ENABLE_SSE42)
    file (GLOB LIBRARY_SRC
           ${LIBRARY_SRC}
           ${CMAKE_CURRENT_SOURCE_DIR}/cpu_x86_sse42/*.cpp
          )
    file (GLOB LIBRARY_HEADERS
           ${LIBRARY_HEADERS}
           ${CMAKE_CURRENT_SOURCE_DIR}/cpu_x86_sse42/*.hpp
          )
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cpu_x86_sse42)
    if (WIN32)
        set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/cpu_x86_sse42/blob_transform_sse42.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/cpu_x86_sse42/ie_preprocess_data_sse42.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/cpu_x86_sse42/ie_preprocess_gapi_kernels_sse42.cpp" PROPERTIES COMPILE_FLAGS /arch:SSE2)
    else()
        set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/cpu_x86_sse42/blob_transform_sse42.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/cpu_x86_sse42/ie_preprocess_data_sse42.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/cpu_x86_sse42/ie_preprocess_gapi_kernels_sse42.cpp" PROPERTIES COMPILE_FLAGS -msse4.2)
    endif()
    add_definitions(-DHAVE_SSE=1)
endif()

addVersionDefines(ie_version.cpp CI_BUILD_NUMBER)
addVersionDefines(hetero/hetero_plugin.cpp CI_BUILD_NUMBER)

set (PUBLIC_HEADERS_DIR "${IE_MAIN_SOURCE_DIR}/include")

file (GLOB PUBLIC_HEADERS
       ${PUBLIC_HEADERS_DIR}/*.h
       ${PUBLIC_HEADERS_DIR}/*.hpp
      )

add_definitions(-DIMPLEMENT_INFERENCE_ENGINE_API)

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj

source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${LIBRARY_HEADERS} ${PUBLIC_HEADERS})

# Create shared library file from sources

add_library(${TARGET_NAME} SHARED
            ${LIBRARY_SRC}
            ${LIBRARY_HEADERS}
            ${PUBLIC_HEADERS})
set_ie_threading_interface_for(${TARGET_NAME})

target_link_libraries(${TARGET_NAME} PRIVATE fluid ngraph ade ${INTEL_ITT_LIBS} pugixml PUBLIC ${CMAKE_DL_LIBS})

if(WIN32)
    #To disable min/max macro in windows.h
    target_compile_definitions(${TARGET_NAME} PRIVATE -DNOMINMAX)
endif()

# Properties->C/C++->General->Additional Include Directories
target_include_directories(${TARGET_NAME} PUBLIC ${PUBLIC_HEADERS_DIR}
                                          PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}"
                                                  "${IE_MAIN_SOURCE_DIR}/src/dumper")

target_include_directories(${TARGET_NAME} SYSTEM PRIVATE "${IE_MAIN_SOURCE_DIR}/thirdparty/pugixml/src")
target_include_directories(${TARGET_NAME} SYSTEM PRIVATE "${IE_MAIN_SOURCE_DIR}/thirdparty/ngraph/src")
target_include_directories(${TARGET_NAME} SYSTEM PRIVATE "${IE_MAIN_SOURCE_DIR}/thirdparty/ocv")

if(ENABLE_MKL_DNN)
    target_include_directories(${TARGET_NAME} SYSTEM PRIVATE "${IE_MAIN_SOURCE_DIR}/thirdparty/mkl-dnn/src/cpu/xbyak")
endif()

set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_PDB_NAME ${TARGET_NAME})

# Static library used for unit tests which are always built

add_library(${TARGET_NAME}_s STATIC
            ${LIBRARY_SRC}
            ${LIBRARY_HEADERS}
            ${PUBLIC_HEADERS})
set_ie_threading_interface_for(${TARGET_NAME}_s)

# Properties->C/C++->General->Additional Include Directories
target_include_directories(${TARGET_NAME}_s PUBLIC  "${PUBLIC_HEADERS_DIR}"
                                            PUBLIC  "${CMAKE_CURRENT_SOURCE_DIR}"
                                            PRIVATE "${IE_MAIN_SOURCE_DIR}/src/dumper")

target_include_directories(${TARGET_NAME}_s SYSTEM PRIVATE "${IE_MAIN_SOURCE_DIR}/thirdparty/pugixml/src")
target_include_directories(${TARGET_NAME}_s SYSTEM PRIVATE "${IE_MAIN_SOURCE_DIR}/thirdparty/ngraph/src")
target_include_directories(${TARGET_NAME}_s SYSTEM PRIVATE "${IE_MAIN_SOURCE_DIR}/thirdparty/ocv")

if(ENABLE_MKL_DNN)
  target_include_directories(${TARGET_NAME}_s SYSTEM PRIVATE "${IE_MAIN_SOURCE_DIR}/thirdparty/mkl-dnn/src/cpu/xbyak")
endif()

target_compile_definitions(${TARGET_NAME}_s PUBLIC -DUSE_STATIC_IE)

if(WIN32)
    #To disable min/max macro in windows.h
    target_compile_definitions(${TARGET_NAME}_s PRIVATE -DNOMINMAX)
endif()

set_target_properties(${TARGET_NAME}_s PROPERTIES COMPILE_PDB_NAME ${TARGET_NAME}_s)

target_link_libraries(${TARGET_NAME}_s PRIVATE fluid
                                       PRIVATE ngraph
                                       PRIVATE ade
                                       PUBLIC pugixml
# NOTE: public linkage with ITT is important here! without it binaries that use this library do not
#       get proper compile flags to support ITT which causes hard-to-detect bugs.
                                       PUBLIC ${INTEL_ITT_LIBS})

add_cpplint_target(${TARGET_NAME}_cpplint FOR_TARGETS ${TARGET_NAME})

ie_register_plugins(MAIN_TARGET ${TARGET_NAME}
                    POSSIBLE_PLUGINS clDNNPlugin dliaPlugin GNAPlugin MKLDNNPlugin myriadPlugin)

# export targets
export(TARGETS ${TARGET_NAME} NAMESPACE IE:: FILE "${CMAKE_BINARY_DIR}/targets.cmake")

configure_file(
    "${PROJECT_SOURCE_DIR}/cmake/config.cmake.in"
    "${CMAKE_BINARY_DIR}/InferenceEngineConfig.cmake"
    COPYONLY)

configure_file(
    "${PROJECT_SOURCE_DIR}/cmake/share/InferenceEngineConfig-version.cmake.in"
    "${CMAKE_BINARY_DIR}/InferenceEngineConfig-version.cmake"
    COPYONLY)

# developer package

add_library(xbyak INTERFACE)
target_include_directories(xbyak INTERFACE ${IE_MAIN_SOURCE_DIR}/thirdparty/mkl-dnn/src/cpu/xbyak)

export(TARGETS ${TARGET_NAME} xbyak NAMESPACE IE:: APPEND FILE "${CMAKE_BINARY_DIR}/targets_developer.cmake")

configure_file(
    "${CMAKE_SOURCE_DIR}/cmake/developer_package_config.cmake.in"
    "${CMAKE_BINARY_DIR}/InferenceEngineDeveloperPackageConfig.cmake"
    @ONLY)

configure_file(
    "${CMAKE_SOURCE_DIR}/cmake/share/InferenceEngineConfig-version.cmake.in"
    "${CMAKE_BINARY_DIR}/InferenceEngineDeveloperPackageConfig-version.cmake"
    COPYONLY)
