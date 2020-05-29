# Copyright (C) 2018-2020 Intel Corporation
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

## Enable Models multiple search pathes
message("configuring file: ${CMAKE_CURRENT_BINARY_DIR}/test_model_repo.h")

function(add_helpers target_name)
    add_library(${target_name} STATIC ${HELPERS_SRC})

    target_include_directories(${target_name} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
                                                     "${IE_MAIN_SOURCE_DIR}/src/inference_engine"
                                                     $<TARGET_PROPERTY:inference_engine_ir_reader,INTERFACE_INCLUDE_DIRECTORIES>
                                                     $<TARGET_PROPERTY:inference_engine_lp_transformations,INTERFACE_INCLUDE_DIRECTORIES>
                                                     $<TARGET_PROPERTY:pugixml,INTERFACE_INCLUDE_DIRECTORIES>
                                                     "${IE_MAIN_SOURCE_DIR}/src/vpu/"
                                             PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")

    # TODO: eliminate dependency on samples
    target_include_directories(${target_name} PUBLIC
        "${IE_MAIN_SOURCE_DIR}/samples/common/os/windows")

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
