# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_auto_unit_tests)

set(CI_BUILD_NUMBER "unittest")
add_definitions(-DMULTIUNITTEST)

set(OV_AUTO_UNIT_TESTS_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/auto_unit_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/compile_model_metric_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/compile_model_property_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ctput_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/default_perf_hint_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/device_telemetry_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dynamic_output_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/get_device_list.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/auto_unit_test.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/gmock_plugin.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/mock_log_utils.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/infer_request_schedule_policy_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/key_network_priority_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/life_time_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/log_utils_format_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/log_utils_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/meta_device_check_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/parse_meta_device_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/perf_curve_table_validator_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/release_helper_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime_fallback_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/select_device_failed_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/select_device_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/set_log_level_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/startup_fallback_property_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/stateful_model_test.cpp
    ${OpenVINO_SOURCE_DIR}/src/plugins/auto/src/async_infer_request.cpp
    ${OpenVINO_SOURCE_DIR}/src/plugins/auto/src/async_infer_request.hpp
    ${OpenVINO_SOURCE_DIR}/src/plugins/auto/src/auto_compiled_model.cpp
    ${OpenVINO_SOURCE_DIR}/src/plugins/auto/src/auto_compiled_model.hpp
    ${OpenVINO_SOURCE_DIR}/src/plugins/auto/src/auto_schedule.cpp
    ${OpenVINO_SOURCE_DIR}/src/plugins/auto/src/auto_schedule.hpp
    ${OpenVINO_SOURCE_DIR}/src/plugins/auto/src/common.hpp
    ${OpenVINO_SOURCE_DIR}/src/plugins/auto/src/compiled_model.cpp
    ${OpenVINO_SOURCE_DIR}/src/plugins/auto/src/compiled_model.hpp
    ${OpenVINO_SOURCE_DIR}/src/plugins/auto/src/cumulative_compiled_model.cpp
    ${OpenVINO_SOURCE_DIR}/src/plugins/auto/src/cumulative_compiled_model.hpp
    ${OpenVINO_SOURCE_DIR}/src/plugins/auto/src/cumulative_schedule.cpp
    ${OpenVINO_SOURCE_DIR}/src/plugins/auto/src/cumulative_schedule.hpp
    ${OpenVINO_SOURCE_DIR}/src/plugins/auto/src/infer_request.cpp
    ${OpenVINO_SOURCE_DIR}/src/plugins/auto/src/infer_request.hpp
    ${OpenVINO_SOURCE_DIR}/src/plugins/auto/src/itt.hpp
    ${OpenVINO_SOURCE_DIR}/src/plugins/auto/src/plugin.cpp
    ${OpenVINO_SOURCE_DIR}/src/plugins/auto/src/plugin.hpp
    ${OpenVINO_SOURCE_DIR}/src/plugins/auto/src/plugin_config.cpp
    ${OpenVINO_SOURCE_DIR}/src/plugins/auto/src/plugin_config.hpp
    ${OpenVINO_SOURCE_DIR}/src/plugins/auto/src/schedule.cpp
    ${OpenVINO_SOURCE_DIR}/src/plugins/auto/src/schedule.hpp
    ${OpenVINO_SOURCE_DIR}/src/plugins/auto/src/utils/device_telemetry.cpp
    ${OpenVINO_SOURCE_DIR}/src/plugins/auto/src/utils/device_telemetry.hpp
    ${OpenVINO_SOURCE_DIR}/src/plugins/auto/src/utils/log.cpp
    ${OpenVINO_SOURCE_DIR}/src/plugins/auto/src/utils/log.hpp
    ${OpenVINO_SOURCE_DIR}/src/plugins/auto/src/utils/log_util.hpp
    ${OpenVINO_SOURCE_DIR}/src/plugins/auto/src/utils/non_copyable.hpp
    ${OpenVINO_SOURCE_DIR}/src/plugins/auto/src/utils/singleton.hpp
    ${OpenVINO_SOURCE_DIR}/src/plugins/auto/src/utils/time_utils.cpp
    ${OpenVINO_SOURCE_DIR}/src/plugins/auto/src/utils/time_utils.hpp
)

ov_add_test_target(
        NAME ${TARGET_NAME}
        SOURCES
            ${OV_AUTO_UNIT_TESTS_SRC}
        INCLUDES
            ${OpenVINO_SOURCE_DIR}/src/plugins/auto
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${OpenVINO_SOURCE_DIR}/src/plugins/auto/src
        LINK_LIBRARIES
            unit_test_utils
        DEPENDENCIES
            openvino_template_extension
            mock_engine
        CHECK_SOURCES_LISTED
        LABELS
            OV UNIT MULTI AUTO
)

ov_add_version_defines(${OpenVINO_SOURCE_DIR}/src/plugins/auto/src/plugin.cpp ${TARGET_NAME})

if(TARGET openvino::ipf_client_api)
    target_compile_definitions(${TARGET_NAME} PRIVATE OV_AUTO_ENABLE_IPF)
    target_link_libraries(${TARGET_NAME} PRIVATE openvino::ipf_client_api)
    target_include_directories(${TARGET_NAME} SYSTEM PRIVATE
        $<TARGET_PROPERTY:nlohmann_json::nlohmann_json,INTERFACE_INCLUDE_DIRECTORIES>)
endif()

ov_set_threading_interface_for(${TARGET_NAME})
