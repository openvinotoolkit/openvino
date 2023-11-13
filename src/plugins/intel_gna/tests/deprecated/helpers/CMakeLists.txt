# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ieTestHelpers)

file(GLOB HELPERS_SRC
       ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

file (GLOB HELPERS_INCLUDES
       ${CMAKE_CURRENT_SOURCE_DIR}/*.h
       ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
      )

function(add_helpers target_name)
    add_library(${target_name} STATIC ${HELPERS_SRC})

    target_include_directories(${target_name}
        PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
                $<TARGET_PROPERTY:ov_gna_func_tests,INCLUDE_DIRECTORIES>
                $<TARGET_PROPERTY:ie_samples_utils,INTERFACE_INCLUDE_DIRECTORIES>
                $<TARGET_PROPERTY:inference_engine_legacy,INTERFACE_INCLUDE_DIRECTORIES>
                $<TARGET_PROPERTY:inference_engine_plugin_api,INTERFACE_INCLUDE_DIRECTORIES>
        PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
endfunction()

add_helpers(${TARGET_NAME})

target_link_libraries(${TARGET_NAME} PUBLIC common_test_utils
                                     PRIVATE inference_engine_legacy)

add_helpers(${TARGET_NAME}_s USE_STATIC_IE)

target_link_libraries(${TARGET_NAME}_s PUBLIC common_test_utils_s
                                       PRIVATE inference_engine_legacy_s)
