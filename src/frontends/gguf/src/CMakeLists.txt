# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# Native GGUF files and supplied GgufDecoder graphs use the same op translators.
# The linkable frontend requires explicit selection by framework name.
# dev_api/ installs the extension API; it does not promise compatibility across releases.
set(GGUF_FRONTEND_DEV_API "${CMAKE_CURRENT_SOURCE_DIR}/../dev_api")

ov_add_frontend(NAME gguf
                LINKABLE_FRONTEND
                FILEDESCRIPTION "FrontEnd to convert GGUF models"
                LINK_LIBRARIES openvino::util openvino::core::dev)

ov_build_target_faster(openvino_gguf_frontend PCH)

# Expose builder API headers to consumers of openvino::frontend::gguf in build and install trees.
target_include_directories(openvino_gguf_frontend
                           PUBLIC $<BUILD_INTERFACE:${GGUF_FRONTEND_DEV_API}>)

install(DIRECTORY "${GGUF_FRONTEND_DEV_API}/openvino"
        DESTINATION ${FRONTEND_INSTALL_INCLUDE}
        COMPONENT ${OV_CPACK_COMP_CORE_DEV}
        ${OV_CPACK_COMP_CORE_DEV_EXCLUDE_ALL}
        FILES_MATCHING PATTERN "*.hpp")

# ov::parallel_for (quant dequant) needs the threading (TBB) include directories.
ov_set_threading_interface_for(openvino_gguf_frontend)
