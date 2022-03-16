// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/evaluate_extension.hpp"

#include <iostream>
#include <unordered_map>

#include "openvino/runtime/remote_tensor.hpp"

ov::EvaluateExtension::~EvaluateExtension() = default;

bool ov::EvaluateExtension::is_host_tensors(const ov::TensorVector& tensors) const {
    for (const auto& tensor : tensors) {
        if (tensor.is<ov::RemoteTensor>())
            return false;
    }
    return true;
}
