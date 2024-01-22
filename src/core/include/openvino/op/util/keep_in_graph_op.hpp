// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "openvino/op/util/framework_node.hpp"

namespace ov {
namespace op {
namespace util {

class OPENVINO_API KeepInGraphOp : public ov::op::util::FrameworkNode {
public:
    OPENVINO_OP("KeepInGraphOp", "util");

    KeepInGraphOp() = default;

    KeepInGraphOp(const OutputVector& inputs, size_t num_outputs, const std::string& op_type_name)
        : ov::op::util::FrameworkNode(inputs, std::max(num_outputs, size_t(1))),
          m_op_type(op_type_name) {}

    std::string get_op_type() const {
        return m_op_type;
    }

protected:
    std::string m_op_type;
};
}  // namespace util
}  // namespace op
}  // namespace ov
