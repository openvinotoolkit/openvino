# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME GnaFunctionalTests)

file(GLOB TEST_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instance/io_blob_tests/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instance/ie_class/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instance/input_tests/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instance/lstm/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instance/single_layer_tests/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/backward_compatibility/*.cpp)

list(APPEND DEPENDENCIES
        HeteroPlugin
        GNAPlugin)

if(ENABLE_MKL_DNN)
    list(APPEND DEPENDENCIES
        MKLDNNPlugin)
endif()

add_executable(${TARGET_NAME} ${TEST_SRC} ${TEST_INCLUDE})

if(GNA_LIBRARY_VERSION STREQUAL "GNA2")
    set(GNA_LIBRARY_VERSION_NUMBER 2)
else()
    set(GNA_LIBRARY_VERSION_NUMBER 1)
endif()

target_compile_definitions(${TARGET_NAME}
        PRIVATE
            USE_GNA=ON
            GNA_LIB_VER=${GNA_LIBRARY_VERSION_NUMBER}
        PUBLIC ${ARGV}
            DATA_PATH=\"${DATA_PATH}\"
            MODELS_PATH=\"${MODELS_PATH}\")

target_link_libraries(${TARGET_NAME}
        PRIVATE
            IESharedTests
            funcTestUtils
        )

target_include_directories(${TARGET_NAME}
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_dependencies(${TARGET_NAME} ${DEPENDENCIES})

add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})
set_property(TEST ${TARGET_NAME} PROPERTY LABELS GNA)

if(GNA_LIBRARY_VERSION STREQUAL "GNA1")
    target_compile_definitions(${TARGET_NAME} PRIVATE GNA1_LIB)
endif()
