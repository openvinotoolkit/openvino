# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(NOT ENABLE_LLVM_BACKEND)
    return()
endif()

# Add lib/cmake/mlir to CMAKE_PREFIX_PATH to find mlir or use -D CMAKE_PREFIX_PATH=$MLIR_BINARY_PKG_DIR/lib/cmake/mlir
find_package(MLIR REQUIRED CONFIG)
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})

message(STATUS "Found MLIR_LIBS: ${MLIR_LIBS}")
get_target_property(MLIRExecutionEngine_INCLUDE_DIRS MLIRExecutionEngine INCLUDE_DIRECTORIES)
get_target_property(MLIRJitRunner_INCLUDE_DIRS MLIRJitRunner INCLUDE_DIRECTORIES)
message(STATUS "MLIRExecutionEngine includes ${MLIRExecutionEngine_INCLUDE_DIRS}")
message(STATUS "MLIRJitRunner includes ${MLIRJitRunner_INCLUDE_DIRS}")

if(TARGET MLIRExecutionEngine)
    message(STATUS "MLIRExecutionEngine exists")
else()
    message(STATUS "MLIRExecutionEngine does not exist")
endif()

set(TARGET_NAME "npu_mlir_runtime")

file(GLOB_RECURSE SOURCES *.cpp *.hpp *.h)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_library(${TARGET_NAME} SHARED ${SOURCES})

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
        openvino::npu_al
        openvino::npu_zero_utils
)

target_include_directories(${TARGET_NAME}
    PUBLIC
        $<TARGET_PROPERTY:LevelZero::Headers,INTERFACE_INCLUDE_DIRECTORIES>
        $<TARGET_PROPERTY:LevelZero::NPUExt,INTERFACE_INCLUDE_DIRECTORIES>
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