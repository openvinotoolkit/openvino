# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# The GGML frontend is an MVP port of llama.cpp's ggml-openvino frontend.  It is built
# only when ENABLE_OV_GGML_FRONTEND is ON and is marked SKIP_INSTALL so it is never
# shipped in the OpenVINO release package (it is for development / testing only).
ov_add_frontend(NAME ggml
                LINKABLE_FRONTEND
                SKIP_INSTALL
                FILEDESCRIPTION "FrontEnd to convert GGML/GGUF models"
                LINK_LIBRARIES openvino::util openvino::core::dev)

ov_build_target_faster(openvino_ggml_frontend PCH)
