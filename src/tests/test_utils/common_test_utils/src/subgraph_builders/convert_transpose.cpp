// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/subgraph_builders/convert_transpose.hpp"

#include "common_test_utils/node_builders/convolution.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/transpose.hpp"

namespace ov {
namespace test {
namespace utils {
std::shared_ptr<ov::Model> make_convert_transpose(ov::Shape input_shape,
                                                  std::vector<size_t> input_order,
                                                  ov::element::Type type) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(type, input_shape)};
    params.front()->set_friendly_name("Param_1");
    params.front()->output(0).get_tensor().set_names({"data"});

    const auto order = ov::op::v0::Constant::create(element::i32, {input_order.size()}, input_order);

    auto convert = std::make_shared<ov::op::v0::Convert>(params.front(), type);
    convert->set_friendly_name("convert");

    auto transpose = std::make_shared<ov::op::v1::Transpose>(convert, order);
    transpose->set_friendly_name("transpose");

    auto result = std::make_shared<ov::op::v0::Result>(transpose);
    result->set_friendly_name("result");

    std::shared_ptr<ov::Model> model =
        std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{params});
    model->set_friendly_name("ConvertTranspose");
    return model;
}
}  // namespace utils
}  // namespace test
}  // namespace ov