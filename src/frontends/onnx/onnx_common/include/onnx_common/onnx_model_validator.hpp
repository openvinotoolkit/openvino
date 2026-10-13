// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <istream>

namespace ov::frontend::onnx::common {

bool is_valid_model(std::istream& model);

}  // namespace ov::frontend::onnx::common
