// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "openvino/core/extension.hpp"

// Core discovers these factories after loading the plugin, preserving the GPU
// library owner through SOExtension. The registry follows primitive registration
// so newly supported internal operation types do not require a second manual list.
OPENVINO_CREATE_EXTENSIONS(ov::intel_gpu::ProgramBuilder::get_shape_inference_extensions());
