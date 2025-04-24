// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/convolution.hpp"
#include "common_test_utils/subgraph_builders/multiple_input_outpput_double_concat.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

namespace ov {
namespace test {
namespace utils {
std::shared_ptr<ov::Model> make_multiple_input_output_double_concat(ov::Shape input_shape, ov::element::Type type) {
    auto param1 = std::make_shared<ov::op::v0::Parameter>(type, input_shape);
    param1->set_friendly_name("param1");
    param1->output(0).get_tensor().set_names({"data1"});

    auto param2 = std::make_shared<ov::op::v0::Parameter>(type, input_shape);
    param2->set_friendly_name("param2");
    param2->output(0).get_tensor().set_names({"data2"});

    auto concat1 = std::make_shared<ov::op::v0::Concat>(OutputVector{param1, param2}, 1);
    concat1->set_friendly_name("concat_op1");
    concat1->output(0).get_tensor().set_names({"concat1"});

    auto result1 = std::make_shared<ov::op::v0::Result>(concat1);
    result1->set_friendly_name("result1");

    auto concat2 = std::make_shared<ov::op::v0::Concat>(OutputVector{concat1, param2}, 1);
    concat2->set_friendly_name("concat_op2");
    concat2->output(0).get_tensor().set_names({"concat2"});

    auto result2 = std::make_shared<ov::op::v0::Result>(concat2);
    result2->set_friendly_name("result2");

    auto model = std::make_shared<ov::Model>(ov::ResultVector{result1, result2}, ov::ParameterVector{param1, param2});
    model->set_friendly_name("makeMultipleInputOutputDoubleConcat");
    return model;
}
}  // namespace utils
}  // namespace test
}  // namespace ov