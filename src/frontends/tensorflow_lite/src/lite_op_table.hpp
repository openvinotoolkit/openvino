// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <map>
#include <string>

#include "openvino/core/node_vector.hpp"
#include "openvino/frontend/tensorflow/node_context.hpp"
#include "openvino_conversions.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace tensorflow_lite {
namespace op {
std::map<std::string, CreatorFunction> get_supported_ops();

#define OP_CONVERTER(op) OutputVector op(const NodeContext& node)


}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov