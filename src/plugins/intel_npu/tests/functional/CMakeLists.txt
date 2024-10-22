# Copyright (C) 2018-2024 Intel Corporation SPDX-License-Identifier: Apache-2.0
#

if(ENABLE_LTO)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()

if(SUGGEST_OVERRIDE_SUPPORTED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-suggest-override")
endif()

set(TARGET_NAME ov_npu_func_tests)

file(GLOB SOURCES
   "${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/src/backend/*.cpp"
   "${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/src/compiler_adapter/*.cpp"
   "${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/include/intel_npu/backend/*.hpp"
   "${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/include/intel_npu/compiler_adapter/*.hpp")
add_library(npuUtils STATIC ${SOURCES})
set(NPU_UTILS_LINK_LIBRARIES
        openvino_npu_zero_result_parser
        openvino::npu_al
        openvino::npu_common
        openvino::npu_logger_utils
        openvino::runtime
        openvino::runtime::dev
        ze_loader)

target_compile_definitions(npuUtils PRIVATE IMPLEMENT_OPENVINO_RUNTIME_PLUGIN)
target_link_libraries(npuUtils PRIVATE ${NPU_UTILS_LINK_LIBRARIES})
target_include_directories(npuUtils PUBLIC "${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/include/")

ov_add_test_target(
  NAME
  ${TARGET_NAME}
  ROOT
  ${CMAKE_CURRENT_SOURCE_DIR}
  EXCLUDED_SOURCE_PATHS
  INCLUDES
  ${CMAKE_CURRENT_SOURCE_DIR}
  "${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/include/"
  "${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances"
  "${CMAKE_CURRENT_SOURCE_DIR}/behavior"
  "${CMAKE_CURRENT_SOURCE_DIR}/internal"
  DEPENDENCIES
  openvino_intel_npu_plugin
  LINK_LIBRARIES
  npuUtils
  openvino_npu_zero_result_parser
  openvino::func_test_utils
  openvino::funcSharedTests
  openvino::format_reader
  openvino::reference
  openvino::runtime
  openvino::npu_common
  openvino::npu_al)

if(WIN32)
    if(ENABLE_DX12)
        target_compile_definitions(${TARGET_NAME} PRIVATE ENABLE_DX12)
        target_link_libraries(${TARGET_NAME} PRIVATE d3d12 dxcore)
    endif()
endif()


set_target_properties(
  ${TARGET_NAME} PROPERTIES FOLDER ${CMAKE_CURRENT_SOURCE_DIR} CXX_STANDARD 17)

if(MSVC)
  # Enforce standards conformance on MSVC
  target_compile_options(${TARGET_NAME} PRIVATE /Zc:preprocessor)
endif()

target_compile_definitions(${TARGET_NAME} PRIVATE CI_BUILD_NUMBER=\"mock_version\")

install(
  TARGETS ${TARGET_NAME}
  RUNTIME DESTINATION tests
          COMPONENT tests
          EXCLUDE_FROM_ALL)
