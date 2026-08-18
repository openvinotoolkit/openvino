# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# LINKABLE_FRONTEND (no SKIP_INSTALL): installs the library + headers alongside the other
# frontends so a direct linker -- the llama.cpp ggml-openvino backend -- can link
# openvino::frontend::gguf and feed it a live GgufDecoder.
ov_add_frontend(NAME gguf
                LINKABLE_FRONTEND
                FILEDESCRIPTION "FrontEnd to convert GGUF models"
                LINK_LIBRARIES openvino::util openvino::core::dev openvino::offline_transformations)

ov_build_target_faster(openvino_gguf_frontend PCH)

# ov::parallel_for (quant dequant) needs the threading (TBB) include directories.
ov_set_threading_interface_for(openvino_gguf_frontend)
