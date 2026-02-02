# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME openvino_shutdown)

# Use OBJECT instead of STATIC to ensure DLLMain is invoked when the shared library is unloaded
add_library(${TARGET_NAME} OBJECT shutdown.cpp)

add_library(openvino::shutdown ALIAS ${TARGET_NAME})
target_link_libraries(${TARGET_NAME} PRIVATE openvino::itt)
set_target_properties(${TARGET_NAME} PROPERTIES EXPORT_NAME openvino_shutdown)

install(TARGETS ${TARGET_NAME} EXPORT OpenVINOTargets
        COMPONENT ${OV_CPACK_COMP_CORE}
        ${OV_CPACK_COMP_CORE_EXCLUDE_ALL})
