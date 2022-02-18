// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
/// Root of nodes that can be sink nodes
class OPENVINO_API Sink : public Op {
public:
    ~Sink() override = 0;
    OPENVINO_OP("Sink");
    BWDCMP_RTTI_DECLARATION;

protected:
    Sink() : Op() {}

    explicit Sink(const OutputVector& arguments) : Op(arguments) {}
};
}  // namespace op
using SinkVector = std::vector<std::shared_ptr<op::Sink>>;
}  // namespace ov
