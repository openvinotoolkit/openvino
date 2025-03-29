// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/extremum.hpp"

#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"

namespace ov {
namespace test {
namespace utils {
std::shared_ptr<ov::Node> make_extremum(const ov::Output<Node>& in0,
                                        const ov::Output<Node>& in1,
                                        ov::test::utils::MinMaxOpType extremum_type) {
    switch (extremum_type) {
    case ov::test::utils::MinMaxOpType::MINIMUM:
        return std::make_shared<ov::op::v1::Minimum>(in0, in1);
    case ov::test::utils::MinMaxOpType::MAXIMUM:
        return std::make_shared<ov::op::v1::Maximum>(in0, in1);
    default:
        throw std::runtime_error("Incorrect type of Extremum operation");
    }
}
}  // namespace utils
}  // namespace test
}  // namespace ov
