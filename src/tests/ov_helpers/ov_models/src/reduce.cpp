// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <vector>

#include "common_test_utils/test_enums.hpp"
#include "openvino/core/node.hpp"
#include "openvino/op/reduce_l1.hpp"
#include "openvino/op/reduce_l2.hpp"
#include "openvino/op/reduce_logical_and.hpp"
#include "openvino/op/reduce_logical_or.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reduce_sum.hpp"

namespace ngraph {
namespace builder {
std::shared_ptr<ov::Node> makeReduce(const ov::Output<ov::Node>& data,
                                     const ov::Output<ov::Node>& axes,
                                     bool keepDims,
                                     ov::test::utils::ReductionType reductionType) {
    switch (reductionType) {
    case ov::test::utils::Mean:
        return std::make_shared<ov::op::v1::ReduceMean>(data, axes, keepDims);
    case ov::test::utils::Max:
        return std::make_shared<ov::op::v1::ReduceMax>(data, axes, keepDims);
    case ov::test::utils::Min:
        return std::make_shared<ov::op::v1::ReduceMin>(data, axes, keepDims);
    case ov::test::utils::Prod:
        return std::make_shared<ov::op::v1::ReduceProd>(data, axes, keepDims);
    case ov::test::utils::Sum:
        return std::make_shared<ov::op::v1::ReduceSum>(data, axes, keepDims);
    case ov::test::utils::LogicalOr:
        return std::make_shared<ov::op::v1::ReduceLogicalOr>(data, axes, keepDims);
    case ov::test::utils::LogicalAnd:
        return std::make_shared<ov::op::v1::ReduceLogicalAnd>(data, axes, keepDims);
    case ov::test::utils::L1:
        return std::make_shared<ov::op::v4::ReduceL1>(data, axes, keepDims);
    case ov::test::utils::L2:
        return std::make_shared<ov::op::v4::ReduceL2>(data, axes, keepDims);
    default:
        throw std::runtime_error("Can't create layer for this reduction type");
    }
}
}  // namespace builder
}  // namespace ngraph
