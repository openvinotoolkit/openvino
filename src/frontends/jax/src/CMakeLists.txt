# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ov_add_frontend(NAME jax
                LINKABLE_FRONTEND
                DISABLE_CPP_INSTALL
                FILEDESCRIPTION "FrontEnd to load and convert JAX/Flax models"
                LINK_LIBRARIES openvino::util openvino::core::dev)

ov_build_target_faster(openvino_jax_frontend PCH)
