# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# [cmake:extension]
set(CMAKE_CXX_STANDARD 11)

set(TARGET_NAME "template_extension")

find_package(ngraph REQUIRED OPTIONAL_COMPONENTS onnx_importer)
find_package(InferenceEngine REQUIRED)
find_package(OpenCV QUIET COMPONENTS core)

set(SRC cpu_kernel.cpp extension.cpp op.cpp)

if (OpenCV_FOUND)
    set(SRC ${SRC} fft_kernel.cpp fft_op.cpp)
endif()

set(SRC ${SRC} unpool_kernel.cpp unpool_op.cpp)

add_library(${TARGET_NAME} MODULE ${SRC})

if (OpenCV_FOUND)
    target_compile_definitions(${TARGET_NAME} PRIVATE OPENCV_IMPORT_ENABLED)
    target_link_libraries(${TARGET_NAME} PRIVATE opencv_core)
endif()

target_compile_definitions(${TARGET_NAME} PRIVATE IMPLEMENT_INFERENCE_EXTENSION_API)
target_link_libraries(${TARGET_NAME} PRIVATE IE::inference_engine
                                             ${NGRAPH_LIBRARIES})

if (ngraph_onnx_importer_FOUND)
    target_link_libraries(${TARGET_NAME} PRIVATE ${ONNX_IMPORTER_LIBRARIES})
    target_compile_definitions(${TARGET_NAME} PRIVATE NGRAPH_ONNX_IMPORT_ENABLED)
endif()

if (IE_CPACK_RUNTIME_PATH)
    ie_cpack_add_component(${TARGET_NAME} REQUIRED DEPENDS core)
    install(TARGETS ${TARGET_NAME}
            RUNTIME DESTINATION ${IE_CPACK_RUNTIME_PATH} COMPONENT ${TARGET_NAME}
            ARCHIVE DESTINATION ${IE_CPACK_ARCHIVE_PATH} COMPONENT ${TARGET_NAME}
            LIBRARY DESTINATION ${IE_CPACK_LIBRARY_PATH} COMPONENT ${TARGET_NAME})
endif()
# [cmake:extension]

# Enable code style check
file(GLOB_RECURSE template_extension_src "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp")
add_clang_format_target(${TARGET_NAME}_clang FOR_SOURCES ${template_extension_src})
