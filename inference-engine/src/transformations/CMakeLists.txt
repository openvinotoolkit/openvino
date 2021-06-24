# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "inference_engine_transformations")

file(GLOB_RECURSE LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE PUBLIC_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)

set(PUBLIC_HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj

source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${PUBLIC_HEADERS})

# Create shared library

add_library(${TARGET_NAME} SHARED ${LIBRARY_SRC} ${PUBLIC_HEADERS})

ie_faster_build(${TARGET_NAME}
    UNITY
    PCH PRIVATE "src/precomp.hpp"
)

ie_add_vs_version_file(NAME ${TARGET_NAME}
                       FILEDESCRIPTION "Inference Engine Transformations library")

target_link_libraries(${TARGET_NAME} PUBLIC ${NGRAPH_LIBRARIES}
                                     PRIVATE ${NGRAPH_REF_LIBRARIES} openvino::itt ngraph::builder pugixml)

target_include_directories(${TARGET_NAME} PUBLIC $<BUILD_INTERFACE:${PUBLIC_HEADERS_DIR}>
                                          PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")

add_cpplint_target(${TARGET_NAME}_cpplint FOR_TARGETS ${TARGET_NAME})

ie_add_api_validator_post_build_step(TARGET ${TARGET_NAME})

ie_mark_target_as_cc(${TARGET_NAME})

# LTO

set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

# developer package

openvino_developer_export_targets(COMPONENT inference_engine TARGETS ${TARGET_NAME})

# install

install(TARGETS ${TARGET_NAME} EXPORT InferenceEngineTargets
        RUNTIME DESTINATION ${IE_CPACK_RUNTIME_PATH} COMPONENT core
        ARCHIVE DESTINATION ${IE_CPACK_ARCHIVE_PATH} COMPONENT core
        LIBRARY DESTINATION ${IE_CPACK_LIBRARY_PATH} COMPONENT core)
