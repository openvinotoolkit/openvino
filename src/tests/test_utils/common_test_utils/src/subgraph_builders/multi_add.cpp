// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/include/common_test_utils/subgraph_builders/multi_add.hpp"

#include "openvino/op/ops.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

namespace ov {
namespace test {
namespace utils {
std::shared_ptr<ov::Model> make_multi_add(ov::Shape input_shape, ov::element::Type type) {
    auto param = std::make_shared<ov::op::v0::Parameter>(type, input_shape);
    param->set_friendly_name("input");

    auto const_value1 = ov::op::v0::Constant::create(type, input_shape, {1});
    const_value1->set_friendly_name("const_value1");
    auto add1 = std::make_shared<ov::op::v1::Add>(param, const_value1);
    add1->set_friendly_name("add1");

    auto const_value2 = ov::op::v0::Constant::create(type, input_shape, {1});
    const_value2->set_friendly_name("const_value2");
    auto add2 = std::make_shared<ov::op::v1::Add>(param, const_value2);
    add2->set_friendly_name("add2");

    auto add3 = std::make_shared<ov::op::v1::Add>(add1, add2);
    add3->set_friendly_name("add3");

    auto const_value4 = ov::op::v0::Constant::create(type, input_shape, {1});
    const_value4->set_friendly_name("const_value4");
    auto add4 = std::make_shared<ov::op::v1::Add>(add3, const_value4);
    add4->set_friendly_name("add4");

    auto const_value5 = ov::op::v0::Constant::create(type, input_shape, {1});
    const_value5->set_friendly_name("const_value5");
    auto add5 = std::make_shared<ov::op::v1::Add>(add3, const_value5);
    add5->set_friendly_name("add5");

    auto add6 = std::make_shared<ov::op::v1::Add>(add4, add5);
    add6->set_friendly_name("add6");

    auto result = std::make_shared<ov::op::v0::Result>(add6);
    result->set_friendly_name("res");

    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
    model->set_friendly_name("Multiadd");
    return model;
}
}  // namespace utils
}  // namespace test
}  // namespace ov