// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/extension.hpp"

#include "openvino/core/op_extension.hpp"
#include "ov_ops/rms.hpp"

#define OP_EXTENSION(NAME) std::make_shared<ov::OpExtension<NAME>>(),

#define NPU_SUPPORTED_EXTENSIONS OP_EXTENSION(ov::op::internal::RMS)

OPENVINO_CREATE_EXTENSIONS(std::vector<ov::Extension::Ptr>({NPU_SUPPORTED_EXTENSIONS}));
