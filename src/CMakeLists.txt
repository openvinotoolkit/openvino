# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_runtime)

if(ENABLE_LTO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()
add_definitions(-DPROJECT_ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

add_custom_target(ov_runtime_libraries ALL)

add_library(${TARGET_NAME} $<TARGET_OBJECTS:ngraph_obj>)
add_library(openvino::runtime ALIAS ${TARGET_NAME})
set_target_properties(${TARGET_NAME} PROPERTIES EXPORT_NAME runtime)
ie_add_vs_version_file(NAME ${TARGET_NAME} FILEDESCRIPTION "OpenVINO runtime library")

target_include_directories(${TARGET_NAME} PUBLIC $<BUILD_INTERFACE:$<TARGET_PROPERTY:ngraph,INTERFACE_INCLUDE_DIRECTORIES>>)
if(WIN32)
    set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_PDB_NAME ${TARGET_NAME})
endif()

# LTO
set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

ie_mark_target_as_cc(${TARGET_NAME})

# TODO: Fix Wall for core, frontends, tests and  c binding
add_subdirectory(core)
add_subdirectory(frontends)
# TODO: remove ngraph/ngraph.hpp usage
add_subdirectory(core/tests)
add_subdirectory(bindings)

if(CMAKE_COMPILER_IS_GNUCXX)
    ie_add_compiler_flags(-Wall)
    ie_add_compiler_flags(-Wmissing-declarations)
endif()

add_subdirectory(inference)
add_subdirectory(common)


if(NGRAPH_IR_FRONTEND_ENABLE)
    if(BUILD_SHARED_LIBS)
        add_dependencies(ov_runtime_libraries ir_ov_frontend)
    endif()
    # use this one once CVS-69781 is fixed
    # add_dependencies(inference_engine ir_ov_frontend)
endif()
if(NGRAPH_ONNX_FRONTEND_ENABLE)
    add_dependencies(inference_engine onnx_ov_frontend)
endif()

if(NGRAPH_PDPD_FRONTEND_ENABLE)
    add_dependencies(inference_engine paddlepaddle_ov_frontend)
endif()

if(NGRAPH_TF_FRONTEND_ENABLE)
    add_dependencies(inference_engine tensorflow_ov_frontend)
endif()

install(TARGETS ${TARGET_NAME} EXPORT OpenVINOTargets
        RUNTIME DESTINATION ${IE_CPACK_RUNTIME_PATH} COMPONENT core
        ARCHIVE DESTINATION ${IE_CPACK_ARCHIVE_PATH} COMPONENT core
        LIBRARY DESTINATION ${IE_CPACK_LIBRARY_PATH} COMPONENT core
        INCLUDES DESTINATION runtime/include
                             runtime/include/ie)

# Install cmake scripts
install(EXPORT OpenVINOTargets
        FILE OpenVINOTargets.cmake
        NAMESPACE openvino::
        DESTINATION runtime/cmake
        COMPONENT core_dev)
