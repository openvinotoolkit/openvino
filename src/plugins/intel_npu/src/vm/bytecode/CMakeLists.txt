set(TARGET_NAME npu_bytecode)

add_library(${TARGET_NAME} INTERFACE)
add_library(openvino::npu_bytecode ALIAS ${TARGET_NAME})

target_include_directories(${TARGET_NAME} INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# target_link_libraries(${TARGET_NAME} INTERFACE VPUXLoader)

ov_add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})

ov_developer_package_export_targets(TARGET openvino::npu_bytecode
                                    INSTALL_INCLUDE_DIRECTORIES
                                        ${CMAKE_CURRENT_SOURCE_DIR}/include/)
