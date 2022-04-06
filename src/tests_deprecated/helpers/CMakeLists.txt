# Copyright (C) 2018-2022 Intel Corporation
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

## Enable Models multiple search paths
message("configuring file: ${CMAKE_CURRENT_BINARY_DIR}/test_model_repo.h")

function(add_helpers target_name)
    add_library(${target_name} STATIC ${HELPERS_SRC})

    target_include_directories(${target_name}
        PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
                $<TARGET_PROPERTY:ie_samples_utils,INTERFACE_INCLUDE_DIRECTORIES>
                $<TARGET_PROPERTY:pugixml::static,INTERFACE_INCLUDE_DIRECTORIES>
                $<TARGET_PROPERTY:inference_engine_legacy,INTERFACE_INCLUDE_DIRECTORIES>
                $<TARGET_PROPERTY:inference_engine_plugin_api,INTERFACE_INCLUDE_DIRECTORIES>
        PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")

    set_property(TARGET ${target_name} PROPERTY COMPILE_PDB_NAME ${target_name})

    # add_cpplint_target(${target_name}_cpplint FOR_TARGETS ${target_name})
endfunction()

add_helpers(${TARGET_NAME})

target_link_libraries(${TARGET_NAME} PUBLIC commonTestUtils)

add_helpers(${TARGET_NAME}_s USE_STATIC_IE)

target_link_libraries(${TARGET_NAME}_s PUBLIC commonTestUtils_s)

if (ENABLE_DATA)
    add_dependencies(${TARGET_NAME} data)
    add_dependencies(${TARGET_NAME}_s data)
endif()

# developer package

openvino_developer_export_targets(COMPONENT tests
                                  TARGETS ieTestHelpers ieTestHelpers_s)
