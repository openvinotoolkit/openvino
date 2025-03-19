// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/subgraph_builders/split_concat.hpp"

#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/split.hpp"

namespace ov {
namespace test {
namespace utils {
std::shared_ptr<ov::Model> make_split_concat(ov::Shape input_shape, ov::element::Type type) {
    auto param1 = std::make_shared<ov::op::v0::Parameter>(type, input_shape);
    param1->set_friendly_name("Param1");
    param1->output(0).get_tensor().set_names({"data1"});

    auto axis_node = ov::op::v0::Constant::create(element::i64, Shape{}, {1});
    auto split = std::make_shared<ov::op::v1::Split>(param1, axis_node, 2);
    split->set_friendly_name("Split");
    split->output(0).get_tensor().set_names({"tensor_split_1"});
    split->output(1).get_tensor().set_names({"tensor_split_2"});

    auto concat = std::make_shared<ov::op::v0::Concat>(OutputVector{split->output(0), split->output(1)}, 1);
    concat->set_friendly_name("Concat_op");
    concat->output(0).get_tensor().set_names({"Concat"});

    auto result = std::make_shared<ov::op::v0::Result>(concat);
    result->set_friendly_name("Result");

    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param1});
    model->set_friendly_name("SplitConcat");
    return model;
}
}  // namespace utils
}  // namespace test
}  // namespace ov