# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(ENABLE_LTO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()

set(TARGET_NAME ov_npu_func_tests)
set(EXCLUDED_FUNC_TESTS_DIR "")
set(OPTIONAL_FUNC_TESTS_INCLUDES "")
set(OPTIONAL_FUNC_TESTS_LIBS "")

if(ENABLE_DRIVER_COMPILER_ADAPTER)
    list(APPEND OPTIONAL_FUNC_TESTS_INCLUDES
        "${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/compiler/include"
    )

    list(APPEND OPTIONAL_FUNC_TESTS_LIBS
        openvino_npu_driver_compiler_adapter
    )
endif()

ov_add_test_target(
    NAME ${TARGET_NAME}
    ROOT ${CMAKE_CURRENT_SOURCE_DIR}
    EXCLUDED_SOURCE_PATHS
        ${EXCLUDED_FUNC_TESTS_DIR}
    INCLUDES
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${OPTIONAL_FUNC_TESTS_INCLUDES}
        "${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances"
        "${CMAKE_CURRENT_SOURCE_DIR}/behavior"
        "${CMAKE_CURRENT_SOURCE_DIR}/internal"
    LINK_LIBRARIES
        ${OPTIONAL_FUNC_TESTS_LIBS}
        openvino::func_test_utils
        openvino::funcSharedTests
        openvino::format_reader
        openvino::reference
        openvino::runtime
        openvino::npu_al
)

set_target_properties(${TARGET_NAME} PROPERTIES
                      FOLDER ${CMAKE_CURRENT_SOURCE_DIR}
                      CXX_STANDARD 17)

if(MSVC)
    # Enforce standards conformance on MSVC
    target_compile_options(${TARGET_NAME}
        PRIVATE
            /Zc:preprocessor
    )
endif()

install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION tests
        COMPONENT tests
        EXCLUDE_FROM_ALL
)
