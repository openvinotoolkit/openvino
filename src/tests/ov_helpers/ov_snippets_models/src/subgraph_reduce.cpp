// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_reduce.hpp"

#include <snippets/op/subgraph.hpp>

#include "openvino/opsets/opset1.hpp"
#include "common_test_utils/data_utils.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/node_builders/reduce.hpp"
#include "openvino/op/constant.hpp"

namespace ov {
namespace test {
namespace snippets {
std::shared_ptr<ov::Model> ReduceFunction::initOriginal() const {
    auto data = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto constant = ov::op::v0::Constant::create(ov::element::i32, {axes.size()}, axes);
    auto reduce = ov::test::utils::make_reduce(data, constant, keep_dims, reduce_type);
    return std::make_shared<ov::Model>(OutputVector{reduce}, ParameterVector{data});
}
}  // namespace snippets
}  // namespace test
}  // namespace ov
