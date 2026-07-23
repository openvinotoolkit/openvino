# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# LINKABLE_FRONTEND: installed alongside the other frontends so callers (the llama.cpp
# ggml-openvino backend, OpenVINO GenAI) can link openvino::frontend::gguf directly. It is kept
# out of the generic FrontEndManager loading API (FrontEndManager treats "gguf" as hidden); see
# the discoverability note in frontend.cpp.
ov_add_frontend(NAME gguf
                LINKABLE_FRONTEND
                FILEDESCRIPTION "FrontEnd to convert GGUF models"
                LINK_LIBRARIES openvino::util openvino::core::dev openvino::offline_transformations)

ov_build_target_faster(openvino_gguf_frontend PCH)

# ov::parallel_for (quant dequant) needs the threading (TBB) include directories.
ov_set_threading_interface_for(openvino_gguf_frontend)
