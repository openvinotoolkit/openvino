// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<ov::Node> makeMinMax(const ov::Output<Node>& in1,
                                     const ov::Output<Node>& in2,
                                     ov::test::utils::MinMaxOpType opType) {
    switch (opType) {
    case ov::test::utils::MinMaxOpType::MINIMUM:
        return std::make_shared<ov::op::v1::Minimum>(in1, in2);
    case ov::test::utils::MinMaxOpType::MAXIMUM:
        return std::make_shared<ov::op::v1::Maximum>(in1, in2);
    default:
        throw std::logic_error("Unsupported operation type");
    }
}

}  // namespace builder
}  // namespace ngraph
