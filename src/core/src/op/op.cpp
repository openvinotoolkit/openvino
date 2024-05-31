// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/op.hpp"

#include <algorithm>
#include <memory>
#include <sstream>

#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace op {
Op::Op(const ov::OutputVector& args) : Node(args) {}

Op::Op(const OutputVector& arguments, Node::OutputDescriptorFactory make_output_descriptor)
    : Node(arguments, std::move(make_output_descriptor)) {}

}  // namespace op
}  // namespace ov
