// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <map>
#include <string>

#include "node_context.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino_conversions.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
using CreatorFunction = std::function<::ov::OutputVector(const ::ov::frontend::tensorflow::NodeContext&)>;

const std::map<const std::string, const CreatorFunction> get_supported_ops();
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
