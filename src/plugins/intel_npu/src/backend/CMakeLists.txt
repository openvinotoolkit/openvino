# Copyright (C) 2018-2025 Intel Corporation
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
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../compiler_adapter/include>
)

target_link_libraries(${TARGET_NAME}
    PRIVATE
        openvino::npu_al
        openvino::npu_common
)

if(ENABLE_LLVM_BACKEND)
    target_link_libraries(${TARGET_NAME}
        PRIVATE
            MLIRAnalysis
            MLIRBuiltinToLLVMIRTranslation
            MLIRExecutionEngine
            MLIRIR
            MLIRJitRunner
            MLIRLLVMDialect
            MLIRLLVMToLLVMIRTranslation
            MLIRToLLVMIRTranslationRegistration
            MLIRParser
            MLIRTargetLLVMIRExport
            MLIRSupport
            LLVMSupport
            LLVMCore
            LLVMX86CodeGen
            LLVMX86AsmParser
    )

    set(TEST_SOURCE_DIR ${NPU_PLUGIN_SOURCE_DIR}/thirdparty/jit-tools)
    set(TEST_DEST_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
    if(WIN32)
        set(TEST_DEST_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${config})
    endif()

    message(STATUS "dest dir:${TEST_DEST_DIR}     config:${config}")
    add_custom_command(TARGET ${TARGET_NAME}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${TEST_SOURCE_DIR} ${TEST_DEST_DIR}
        COMMENT "Copying jit files to ${TEST_DEST_DIR} directory"
    )
endif()

#
# targets install
#
ov_install_static_lib(${TARGET_NAME} ${NPU_INTERNAL_COMPONENT})
