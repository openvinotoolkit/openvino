# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ov_add_frontend(NAME gguf
                LINKABLE_FRONTEND
                FILEDESCRIPTION "FrontEnd to load and convert GGUF models"
                LINK_LIBRARIES openvino::util openvino::core::dev openvino_transformations)

ov_build_target_faster(openvino_gguf_frontend PCH)
