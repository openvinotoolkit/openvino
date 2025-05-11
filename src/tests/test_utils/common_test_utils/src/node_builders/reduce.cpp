// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/reduce.hpp"

#include "openvino/op/reduce_l1.hpp"
#include "openvino/op/reduce_l2.hpp"
#include "openvino/op/reduce_logical_and.hpp"
#include "openvino/op/reduce_logical_or.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reduce_sum.hpp"

namespace ov {
namespace test {
namespace utils {
std::shared_ptr<ov::Node> make_reduce(const ov::Output<Node>& data,
                                      const ov::Output<Node>& axes,
                                      bool keep_dims,
                                      ov::test::utils::ReductionType reduction_type) {
    switch (reduction_type) {
    case ov::test::utils::Mean:
        return std::make_shared<ov::op::v1::ReduceMean>(data, axes, keep_dims);
    case ov::test::utils::Max:
        return std::make_shared<ov::op::v1::ReduceMax>(data, axes, keep_dims);
    case ov::test::utils::Min:
        return std::make_shared<ov::op::v1::ReduceMin>(data, axes, keep_dims);
    case ov::test::utils::Prod:
        return std::make_shared<ov::op::v1::ReduceProd>(data, axes, keep_dims);
    case ov::test::utils::Sum:
        return std::make_shared<ov::op::v1::ReduceSum>(data, axes, keep_dims);
    case ov::test::utils::LogicalOr:
        return std::make_shared<ov::op::v1::ReduceLogicalOr>(data, axes, keep_dims);
    case ov::test::utils::LogicalAnd:
        return std::make_shared<ov::op::v1::ReduceLogicalAnd>(data, axes, keep_dims);
    case ov::test::utils::L1:
        return std::make_shared<ov::op::v4::ReduceL1>(data, axes, keep_dims);
    case ov::test::utils::L2:
        return std::make_shared<ov::op::v4::ReduceL2>(data, axes, keep_dims);
    default:
        OPENVINO_THROW("Can't create layer for this reduction type");
    }
}
}  // namespace utils
}  // namespace test
}  // namespace ov
