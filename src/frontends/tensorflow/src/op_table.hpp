// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <map>
#include <string>

#include "openvino/core/node_vector.hpp"
#include "openvino_conversions.hpp"
#include "tensorflow_frontend/node_context.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
<<<<<<< HEAD
using CreatorFunction = std::function<::ov::OutputVector(const ov::frontend::tf::NodeContext&)>;
=======
using CreatorFunction = std::function<::ov::OutputVector(const ::ov::frontend::tensorflow::NodeContext&)>;
>>>>>>> upstream/master

const std::map<const std::string, const CreatorFunction> get_supported_ops();
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
