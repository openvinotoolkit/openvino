// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace dft {

ov::Output<ov::Node> make_dft(const ov::Output<ov::Node>& signal,
                              const ov::Output<ov::Node>& length,
                              int64_t axis,
                              bool is_inversed,
                              bool is_one_sided);
}  // namespace  dft
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
