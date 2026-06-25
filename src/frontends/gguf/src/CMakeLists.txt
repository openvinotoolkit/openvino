# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# The GGUF frontend is an MVP port of llama.cpp's ggml-openvino frontend.  It is built
# only when ENABLE_OV_GGUF_FRONTEND is ON and is marked SKIP_INSTALL so it is never
# shipped in the OpenVINO release package (it is for development / testing only).
ov_add_frontend(NAME gguf
                LINKABLE_FRONTEND
                SKIP_INSTALL
                FILEDESCRIPTION "FrontEnd to convert GGUF models"
                LINK_LIBRARIES openvino::util openvino::core::dev)

ov_build_target_faster(openvino_gguf_frontend PCH)
