// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <istream>

namespace ov {
namespace frontend {
namespace onnx {
namespace common {

bool is_valid_model(std::istream& model);

}  // namespace common
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
