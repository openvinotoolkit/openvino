# Copyright (C) 2018 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set (TARGET_NAME "inference_engine")

file (GLOB LIBRARY_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cpp_interfaces/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/shape_infer/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/shape_infer/built-in/*.cpp
      )

file (GLOB LIBRARY_HEADERS
       ${CMAKE_CURRENT_SOURCE_DIR}/*.h
       ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
       ${CMAKE_CURRENT_SOURCE_DIR}/cpp_interfaces/*.hpp
       ${CMAKE_CURRENT_SOURCE_DIR}/shape_infer/*.hpp
       ${CMAKE_CURRENT_SOURCE_DIR}/shape_infer/built-in/*.hpp
       ${CMAKE_CURRENT_SOURCE_DIR}/cpp_interfaces/base/*.hpp
       ${CMAKE_CURRENT_SOURCE_DIR}/cpp_interfaces/impl/*.hpp
       ${CMAKE_CURRENT_SOURCE_DIR}/cpp_interfaces/interface/*.hpp
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
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/cpu_x86_sse42/blob_transform_sse42.cpp PROPERTIES COMPILE_FLAGS -msse4.2)
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/cpu_x86_sse42/ie_preprocess_data_sse42.cpp PROPERTIES COMPILE_FLAGS -msse4.2)
    add_definitions(-DHAVE_SSE=1)
endif()

addVersionDefines(ie_version.cpp CI_BUILD_NUMBER)

set (PUBLIC_HEADERS_DIR "${IE_MAIN_SOURCE_DIR}/include")

file (GLOB PUBLIC_HEADERS
       ${PUBLIC_HEADERS_DIR}/*.h
       ${PUBLIC_HEADERS_DIR}/*.hpp
      )

if(WIN32)
  add_definitions(-DIMPLEMENT_INFERENCE_ENGINE_API)
endif()


# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj

source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${LIBRARY_HEADERS} ${PUBLIC_HEADERS})

# Create shared library file from sources

add_library(${TARGET_NAME} SHARED
            ${LIBRARY_SRC}
            ${LIBRARY_HEADERS}
            ${PUBLIC_HEADERS})


target_link_libraries(${TARGET_NAME} PRIVATE pugixml ade ${CMAKE_DL_LIBS} ${INTEL_ITT_LIBS})

# Properties->C/C++->General->Additional Include Directories
target_include_directories(${TARGET_NAME} PUBLIC ${PUBLIC_HEADERS_DIR}
                                          PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}"
                                                  "${IE_MAIN_SOURCE_DIR}/src/dumper")

target_include_directories(${TARGET_NAME} SYSTEM PRIVATE "${IE_MAIN_SOURCE_DIR}/thirdparty/pugixml/src")
target_include_directories(${TARGET_NAME} SYSTEM PRIVATE "${IE_MAIN_SOURCE_DIR}/thirdparty/ade/sources/ade/include")
if(ENABLE_MKL_DNN)
    target_include_directories(${TARGET_NAME} SYSTEM PRIVATE "${IE_MAIN_SOURCE_DIR}/thirdparty/mkl-dnn/src/cpu/xbyak")
endif()

set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_PDB_NAME ${TARGET_NAME})

# Static library used for unit tests which are always built

add_library(${TARGET_NAME}_s STATIC
            ${LIBRARY_SRC}
            ${LIBRARY_HEADERS}
            ${PUBLIC_HEADERS})

# Properties->C/C++->General->Additional Include Directories
target_include_directories(${TARGET_NAME}_s PUBLIC ${PUBLIC_HEADERS_DIR}
                                          PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}"
                                                  "${IE_MAIN_SOURCE_DIR}/src/dumper")

target_include_directories(${TARGET_NAME}_s SYSTEM PRIVATE "${IE_MAIN_SOURCE_DIR}/thirdparty/pugixml/src")
target_include_directories(${TARGET_NAME}_s SYSTEM PRIVATE "${IE_MAIN_SOURCE_DIR}/thirdparty/ade/sources/ade/include")
if(ENABLE_MKL_DNN)
  target_include_directories(${TARGET_NAME}_s SYSTEM PRIVATE "${IE_MAIN_SOURCE_DIR}/thirdparty/mkl-dnn/src/cpu/xbyak")
endif()

target_compile_definitions(${TARGET_NAME}_s PUBLIC -DUSE_STATIC_IE)

set_target_properties(${TARGET_NAME}_s PROPERTIES COMPILE_PDB_NAME ${TARGET_NAME}_s)

# export targets

export(TARGETS ${TARGET_NAME} NAMESPACE IE:: FILE "${CMAKE_BINARY_DIR}/targets.cmake")

configure_file(
    "${CMAKE_SOURCE_DIR}/cmake/config.cmake.in"
    "${CMAKE_BINARY_DIR}/InferenceEngineConfig.cmake"
    COPYONLY)

configure_file(
    "${CMAKE_SOURCE_DIR}/cmake/share/InferenceEngineConfig-version.cmake"
    "${CMAKE_BINARY_DIR}/InferenceEngineConfig-version.cmake"
    COPYONLY)
