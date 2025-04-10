// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/subgraph_builders/read_concat_split_assign.hpp"

#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/split.hpp"

namespace ov {
namespace test {
namespace utils {
std::shared_ptr<ov::Model> make_read_concat_split_assign(ov::Shape input_shape, ov::element::Type type) {
    ov::ParameterVector parameter{std::make_shared<ov::op::v0::Parameter>(type, input_shape)};
    parameter[0]->set_friendly_name("parameter");

    auto init_const = ov::op::v0::Constant::create(type, input_shape, {0});
    auto read = std::make_shared<ov::op::v3::ReadValue>(init_const, "v0");
    read->set_friendly_name("read");

    std::vector<std::shared_ptr<ov::Node>> args = {parameter[0], read};
    auto conc = std::make_shared<ov::op::v0::Concat>(args, 3);
    conc->set_friendly_name("concat");

    auto res = std::make_shared<ov::op::v0::Result>(conc);
    res->set_friendly_name("result");

    const auto axis = ov::op::v0::Constant::create(element::i64, Shape{}, {3});
    axis->set_friendly_name("axis");

    auto crop = std::make_shared<ov::op::v1::Split>(conc, axis, 2);
    crop->set_friendly_name("split");

    auto assign = std::make_shared<ov::op::v3::Assign>(crop, "v0");
    assign->set_friendly_name("assign");

    std::shared_ptr<ov::Model> model =
        std::make_shared<ov::Model>(ov::ResultVector({res}), ov::SinkVector({assign}), ov::ParameterVector{parameter});
    model->set_friendly_name("ReadConcatSplitAssign");
    return model;
}
}  // namespace utils
}  // namespace test
}  // namespace ov