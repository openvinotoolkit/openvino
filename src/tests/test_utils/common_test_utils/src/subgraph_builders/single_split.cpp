// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/subgraph_builders/single_split.hpp"

#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/split.hpp"

namespace ov {
namespace test {
namespace utils {
std::shared_ptr<ov::Model> make_single_split(ov::Shape input_shape, ov::element::Type type) {
    auto param1 = std::make_shared<ov::op::v0::Parameter>(type, input_shape);
    param1->set_friendly_name("param1");
    param1->output(0).get_tensor().set_names({"data1"});

    auto axis_node = ov::op::v0::Constant::create(element::i64, Shape{}, {1});
    auto split = std::make_shared<ov::op::v1::Split>(param1, axis_node, 2);
    split->set_friendly_name("split");
    split->output(0).get_tensor().set_names({"tensor_split_1"});
    split->output(1).get_tensor().set_names({"tensor_split_2"});

    auto result1 = std::make_shared<ov::op::v0::Result>(split->output(0));
    result1->set_friendly_name("result1");

    auto result2 = std::make_shared<ov::op::v0::Result>(split->output(1));
    result2->set_friendly_name("result2");

    auto model = std::make_shared<ov::Model>(ov::ResultVector{result1, result2}, ov::ParameterVector{param1});
    model->set_friendly_name("SingleSplit");
    return model;
}
}  // namespace utils
}  // namespace test
}  // namespace ov