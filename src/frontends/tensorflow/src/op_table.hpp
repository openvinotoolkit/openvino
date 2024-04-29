// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <map>
#include <string>

#include "openvino/core/node_vector.hpp"
#include "openvino/frontend/tensorflow/node_context.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

const std::map<std::string, CreatorFunction> get_supported_ops();

const std::vector<std::string> get_supported_ops_via_tokenizers();
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
