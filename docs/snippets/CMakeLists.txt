# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ie_docs_snippets)

file(GLOB SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

# remove OpenCL related sources
# TODO: fix compilation of OpenCL files
if(NOT CLDNN__IOCL_ICD_INCDIRS OR TRUE)
    list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/GPU_RemoteBlob_API0.cpp"
                             "${CMAKE_CURRENT_SOURCE_DIR}/GPU_RemoteBlob_API1.cpp"
                             "${CMAKE_CURRENT_SOURCE_DIR}/GPU_RemoteBlob_API2.cpp")
endif()

# remove OpenCV related sources
if (ENABLE_OPENCV)
  find_package(OpenCV QUIET)
else()
  set(OpenCV_FOUND FALSE)
endif()

if(NOT OpenCV_FOUND)
    list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/dldt_optimization_guide5.cpp"
                             "${CMAKE_CURRENT_SOURCE_DIR}/ShapeInference.cpp")
endif()

# ONNX importer related files
if(NOT NGRAPH_ONNX_IMPORT_ENABLE)
    list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/OnnxImporterTutorial0.cpp"
                             "${CMAKE_CURRENT_SOURCE_DIR}/OnnxImporterTutorial1.cpp"
                             "${CMAKE_CURRENT_SOURCE_DIR}/OnnxImporterTutorial2.cpp"
                             "${CMAKE_CURRENT_SOURCE_DIR}/OnnxImporterTutorial3.cpp")
endif()

# remove snippets for deprecated / removed API
list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/Migration_CoreAPI.cpp")

# requires mfxFrameSurface1 and MSS API
list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/dldt_optimization_guide2.cpp"
                         "${CMAKE_CURRENT_SOURCE_DIR}/dldt_optimization_guide3.cpp"
                         "${CMAKE_CURRENT_SOURCE_DIR}/dldt_optimization_guide4.cpp")

# create a static library

add_library(${TARGET_NAME} STATIC ${SOURCES})

if(CLDNN__IOCL_ICD_INCDIRS)
    target_include_directories(${TARGET_NAME} SYSTEM PRIVATE ${CLDNN__IOCL_ICD_INCDIRS})
endif()

if(OpenCV_FOUND)
    target_include_directories(${TARGET_NAME} SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(${TARGET_NAME} PRIVATE opencv_core)
endif()

if(NGRAPH_ONNX_IMPORT_ENABLE)
    target_link_libraries(${TARGET_NAME} PRIVATE onnx_importer)
endif()

if(NOT MSVC)
    target_compile_options(${TARGET_NAME} PRIVATE -Wno-unused-variable)
    if(CMAKE_COMPILER_IS_GNUCXX)
        target_compile_options(${TARGET_NAME} PRIVATE -Wno-unused-but-set-variable)
    endif()
endif()

target_link_libraries(${TARGET_NAME} PRIVATE inference_engine_plugin_api
    ngraph inference_engine_transformations)
