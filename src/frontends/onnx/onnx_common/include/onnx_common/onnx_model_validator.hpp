// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <istream>

namespace ngraph {
namespace onnx_common {

bool is_valid_model(std::istream& model);
}  // namespace onnx_common
}  // namespace ngraph
