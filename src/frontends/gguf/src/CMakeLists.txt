# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# The GGUF frontend converts GGUF models to OpenVINO. It installs the library + headers alongside
# the other frontends (LINKABLE_FRONTEND, no SKIP_INSTALL) so direct linkers -- the llama.cpp
# ggml-openvino backend and OpenVINO GenAI -- can link openvino::frontend::gguf, and it is
# discoverable via FrontEndManager so core.read_model("model.gguf") selects it by the .gguf
# extension + GGUF magic (see supported_impl / get_front_end_data in frontend.cpp). Two ingest
# paths converge on the same op translators: a live GgufDecoder (cgraph path) and a native
# .gguf-file builder.
# dev_api/ carries the builder SDK an out-of-tree ArchitectureExtension is written against (graph
# context, value handles, hparams/tensor views, ModelBuilder). It is a DEVELOPER API: installed and
# linkable, but deliberately not part of the stable public API contract, like frontends/common's.
set(GGUF_FRONTEND_DEV_API "${CMAKE_CURRENT_SOURCE_DIR}/../dev_api")

ov_add_frontend(NAME gguf
                LINKABLE_FRONTEND
                FILEDESCRIPTION "FrontEnd to convert GGUF models"
                LINK_LIBRARIES openvino::util openvino::core::dev)

ov_build_target_faster(openvino_gguf_frontend PCH)

# The builder SDK headers. PUBLIC so a consumer that links openvino::frontend::gguf -- OpenVINO
# GenAI registering an ArchitectureExtension, say -- can include them without extra include paths,
# and installed next to the frontend's own headers so that works from an installed package too.
target_include_directories(openvino_gguf_frontend
                           PUBLIC $<BUILD_INTERFACE:${GGUF_FRONTEND_DEV_API}>)

install(DIRECTORY "${GGUF_FRONTEND_DEV_API}/openvino"
        DESTINATION ${FRONTEND_INSTALL_INCLUDE}
        COMPONENT ${OV_CPACK_COMP_CORE_DEV}
        ${OV_CPACK_COMP_CORE_DEV_EXCLUDE_ALL}
        FILES_MATCHING PATTERN "*.hpp")

# ov::parallel_for (quant dequant) needs the threading (TBB) include directories.
ov_set_threading_interface_for(openvino_gguf_frontend)
