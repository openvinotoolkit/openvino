// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/subgraph_builders/concat_with_params.hpp"

#include "common_test_utils/node_builders/convolution.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

namespace ov {
namespace test {
namespace utils {
std::shared_ptr<ov::Model> make_concat_with_params(ov::Shape input_shape, ov::element::Type type) {
    auto parameter1 = std::make_shared<ov::op::v0::Parameter>(type, input_shape);
    parameter1->set_friendly_name("param1");
    parameter1->output(0).get_tensor().set_names({"data1"});

    auto parameter2 = std::make_shared<ov::op::v0::Parameter>(type, input_shape);
    parameter2->set_friendly_name("param2");
    parameter2->output(0).get_tensor().set_names({"data2"});

    auto concat = std::make_shared<ov::op::v0::Concat>(OutputVector{parameter1, parameter2}, 1);
    concat->set_friendly_name("concat_op");
    concat->output(0).get_tensor().set_names({"concat"});

    auto result = std::make_shared<ov::op::v0::Result>(concat);
    result->set_friendly_name("result");

    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{parameter1, parameter2});
    model->set_friendly_name("SingleConcatWithParams");
    return model;
}
}  // namespace utils
}  // namespace test
}  // namespace ov