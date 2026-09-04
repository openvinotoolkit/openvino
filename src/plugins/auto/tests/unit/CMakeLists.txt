# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_auto_unit_tests)

set(CI_BUILD_NUMBER "unittest")
add_definitions(-DMULTIUNITTEST)

ov_add_test_target(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        ADDITIONAL_SOURCE_DIRS
            ${OpenVINO_SOURCE_DIR}/src/plugins/auto/src
            ${OpenVINO_SOURCE_DIR}/src/plugins/auto/src/utils
        INCLUDES
            ${OpenVINO_SOURCE_DIR}/src/plugins/auto
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${OpenVINO_SOURCE_DIR}/src/plugins/auto/src
        LINK_LIBRARIES
            unit_test_utils
        DEPENDENCIES
            openvino_template_extension
            mock_engine
        LABELS
            OV UNIT MULTI AUTO
)

ov_add_version_defines(${OpenVINO_SOURCE_DIR}/src/plugins/auto/src/plugin.cpp ${TARGET_NAME})

# See src/plugins/auto/CMakeLists.txt for the per-config static/dynamic ClientApi choice.
if(ENABLE_IPF_DEBUG_WITH_RELEASE_CRT)
    set(_ipf_use_dynamic_genex "$<CONFIG:Debug>")
else()
    set(_ipf_use_dynamic_genex "$<BOOL:FALSE>")
endif()

if(TARGET openvino::ipf_client_api_static AND TARGET openvino::ipf_client_api_dynamic)
    target_compile_definitions(${TARGET_NAME} PRIVATE
        OV_AUTO_ENABLE_IPF
        $<${_ipf_use_dynamic_genex}:ClientApi_IMPORTS>)
    target_link_libraries(${TARGET_NAME} PRIVATE
        $<${_ipf_use_dynamic_genex}:openvino::ipf_client_api_dynamic>
        $<$<NOT:${_ipf_use_dynamic_genex}>:openvino::ipf_client_api_static>
        $<${_ipf_use_dynamic_genex}:delayimp.lib>)
    target_include_directories(${TARGET_NAME} SYSTEM PRIVATE
        $<TARGET_PROPERTY:nlohmann_json::nlohmann_json,INTERFACE_INCLUDE_DIRECTORIES>)

    target_link_options(${TARGET_NAME} PRIVATE $<${_ipf_use_dynamic_genex}:/DELAYLOAD:ClientApi.dll>)

    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
        COMMAND "$<${_ipf_use_dynamic_genex}:${CMAKE_COMMAND};-E;copy_if_different;$<TARGET_FILE:ClientApi>;$<TARGET_FILE_DIR:${TARGET_NAME}>>"
        COMMAND_EXPAND_LISTS)
endif()

ov_set_threading_interface_for(${TARGET_NAME})
