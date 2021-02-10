// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <istream>

namespace InferenceEngine {
    // 2 empty structs used for tag dispatch below
    struct onnx_format {};
    struct prototxt_format {};

    bool is_valid_model(std::istream& model, onnx_format);

    bool is_valid_model(std::istream& model, prototxt_format);
} // namespace InferenceEngine
