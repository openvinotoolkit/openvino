// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/subgraph_builders/single_concat_with_constant.hpp"

#include "common_test_utils/node_builders/convolution.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

namespace ov {
namespace test {
namespace utils {
std::shared_ptr<ov::Model> make_single_concat_with_constant(ov::Shape input_shape, ov::element::Type type) {
    ov::ParameterVector parameter{std::make_shared<ov::op::v0::Parameter>(type, input_shape)};
    parameter[0]->set_friendly_name("Param_1");
    parameter[0]->output(0).get_tensor().set_names({"data"});

    auto init_const = ov::op::v0::Constant::create(type, input_shape, {0});

    std::vector<std::shared_ptr<ov::Node>> args = {parameter[0], init_const};
    auto conc = std::make_shared<ov::op::v0::Concat>(args, 3);
    conc->set_friendly_name("concat");

    auto res = std::make_shared<ov::op::v0::Result>(conc);
    res->set_friendly_name("result");

    std::shared_ptr<ov::Model> model =
        std::make_shared<ov::Model>(ov::ResultVector({res}), ov::ParameterVector{parameter});
    model->set_friendly_name("SingleConcatWithConstant");
    return model;
}
}  // namespace utils
}  // namespace test
}  // namespace ov