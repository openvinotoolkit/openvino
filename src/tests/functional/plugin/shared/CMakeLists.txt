# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME funcSharedTests)

set(PUBLIC_HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

set(DEPENDENCIES mock_engine)

list(APPEND LINK_LIBRARIES_PRIVATE
        openvino::util
        openvino::runtime::dev)

if(ENABLE_HETERO)
    list(APPEND DEPENDENCIES openvino_hetero_plugin)
endif()

if(ENABLE_AUTO OR ENABLE_MULTI)
    list(APPEND DEPENDENCIES openvino_auto_plugin)
endif()

if(ENABLE_AUTO_BATCH)
    list(APPEND DEPENDENCIES openvino_auto_batch_plugin)
endif()

if(ENABLE_TEMPLATE)
    list(APPEND DEPENDENCIES openvino_template_plugin)
endif()

if(ENABLE_OV_IR_FRONTEND)
    list(APPEND DEPENDENCIES openvino_ir_frontend)
endif()

if (ENABLE_OV_ONNX_FRONTEND)
    list(APPEND DEPENDENCIES test_model_zoo openvino_onnx_frontend)
    list(APPEND DEFINES TEST_MODELS="${TEST_MODEL_ZOO}/func_tests/models/")
else()
    set(EXCLUDED_SOURCE_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/src/onnx)
endif()

if (TARGET openvino::snippets)
    list(APPEND LINK_LIBRARIES_PRIVATE ov_snippets_models)
else()
    list(APPEND EXCLUDED_SOURCE_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/src/snippets)
endif()

ov_add_target(
        NAME ${TARGET_NAME}
        TYPE STATIC
        ROOT ${PUBLIC_HEADERS_DIR}
        ADDITIONAL_SOURCE_DIRS
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        ADD_CPPLINT
        EXCLUDED_SOURCE_PATHS ${EXCLUDED_SOURCE_PATHS}
        DEFINES ${DEFINES}
        INCLUDES
            PUBLIC
                "$<BUILD_INTERFACE:${PUBLIC_HEADERS_DIR}>"
        LINK_LIBRARIES
            PUBLIC
                openvino::pugixml
                common_test_utils
                func_test_utils
                ov_models   # TODO (vurusovs): delete after migration of NVidia and NPU
                ov_lpt_models
                sharedTestClasses
            PRIVATE
                ${LINK_LIBRARIES_PRIVATE}
        DEPENDENCIES
            ${DEPENDENCIES}
)

ov_build_target_faster(${TARGET_NAME}
    PCH PRIVATE "src/precomp.hpp"
)

if (ENABLE_INTEL_CPU)
    set_source_files_properties(
        "${CMAKE_CURRENT_SOURCE_DIR}/src/behavior/ov_executable_network/get_metric.cpp"
        PROPERTIES COMPILE_DEFINITIONS ENABLE_INTEL_CPU=1)
endif()

# install & export

ov_developer_package_export_targets(TARGET ${TARGET_NAME}
                                    INSTALL_INCLUDE_DIRECTORIES "${PUBLIC_HEADERS_DIR}/")
