// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"

namespace ngraph {
namespace onnx_import {
namespace dft {

// For DFT, IDFT, IRDFT cases, if real signal are provided (with shape [D_0, D_1, ..., D_{N-1}, 1])
// it's needed to fill tensors with zero imaginary part to be aligned with Core ops requirements.
bool try_convert_real_to_complex(ov::Output<ov::Node>& signal);

ov::Output<ov::Node> make_dft(const ov::Output<ov::Node>& signal,
                              const ov::Output<ov::Node>& length,
                              int64_t axis,
                              bool is_inversed,
                              bool is_one_sided);
}  // namespace  dft
}  // namespace onnx_import
}  // namespace ngraph
