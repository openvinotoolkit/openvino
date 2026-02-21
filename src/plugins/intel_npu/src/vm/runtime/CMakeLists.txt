set(TARGET_NAME npu_mlir_runtime_api)

add_library(${TARGET_NAME} INTERFACE)
add_library(openvino::npu_mlir_runtime_api ALIAS ${TARGET_NAME})

target_include_directories(${TARGET_NAME} INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

ov_add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})

ov_developer_package_export_targets(TARGET openvino::npu_mlir_runtime_api
                                    INSTALL_INCLUDE_DIRECTORIES
                                        ${CMAKE_CURRENT_SOURCE_DIR}/include/)


add_subdirectory(level_zero_wrapper)
add_subdirectory(interpreter)
