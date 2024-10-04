# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "openvino_npu_level_zero_backend")

file(GLOB_RECURSE SOURCES *.cpp *.hpp *.h)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_library(${TARGET_NAME} STATIC ${SOURCES})

set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})
ov_add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})

target_compile_definitions(${TARGET_NAME}
    PRIVATE
        IMPLEMENT_OPENVINO_RUNTIME_PLUGIN)

target_include_directories(${TARGET_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(${TARGET_NAME}
    PRIVATE
        openvino::npu_al
        openvino_npu_zero_result_parser
        ze_loader
)

#
# targets install
#
ov_install_static_lib(${TARGET_NAME} ${NPU_INTERNAL_COMPONENT})

if(TARGET ze_loader)
    if(NOT BUILD_SHARED_LIBS)
        # Support link of static runtime in case system does not have ze_loader
        install(TARGETS ze_loader EXPORT OpenVINOTargets
            RUNTIME DESTINATION ${OV_CPACK_RUNTIMEDIR} COMPONENT ${NPU_PLUGIN_COMPONENT}
            ARCHIVE DESTINATION ${OV_CPACK_ARCHIVEDIR} COMPONENT ${NPU_PLUGIN_COMPONENT}
            LIBRARY DESTINATION ${OV_CPACK_LIBRARYDIR} COMPONENT ${NPU_PLUGIN_COMPONENT})

        install(TARGETS utils EXPORT OpenVINOTargets
            RUNTIME DESTINATION ${OV_CPACK_RUNTIMEDIR} COMPONENT ${NPU_PLUGIN_COMPONENT}
            ARCHIVE DESTINATION ${OV_CPACK_ARCHIVEDIR} COMPONENT ${NPU_PLUGIN_COMPONENT}
            LIBRARY DESTINATION ${OV_CPACK_LIBRARYDIR} COMPONENT ${NPU_PLUGIN_COMPONENT})

        # export to local tree to build against static build tree
        export(TARGETS ze_loader NAMESPACE openvino::
            APPEND FILE "${CMAKE_BINARY_DIR}/OpenVINOTargets.cmake")

        export(TARGETS utils NAMESPACE openvino::
            APPEND FILE "${CMAKE_BINARY_DIR}/OpenVINOTargets.cmake")
    endif()

    # Support tests to run with ze_loader
    install(TARGETS ze_loader
        LIBRARY DESTINATION tests COMPONENT tests EXCLUDE_FROM_ALL)
endif()

