// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "openvino/core/node.hpp"

namespace ov {
namespace op {
/// Root of all actual ops
class OPENVINO_API Op : public Node {
protected:
    Op() : Node() {}
    Op(const OutputVector& arguments);

public:
    static const ::ov::Node::type_info_t type_info;
    const ::ov::Node::type_info_t& get_type_info() const override;
};
}  // namespace op
}  // namespace ov
