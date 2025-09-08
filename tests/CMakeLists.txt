# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(layer_tests)
add_subdirectory(model_hub_tests)
add_subdirectory(samples_tests)
add_subdirectory(e2e_tests)

install(FILES requirements_pytorch requirements_tensorflow requirements_onnx requirements_jax
        DESTINATION tests COMPONENT tests EXCLUDE_FROM_ALL)
