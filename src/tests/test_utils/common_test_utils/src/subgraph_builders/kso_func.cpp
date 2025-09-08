// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/subgraph_builders/kso_func.hpp"

#include "common_test_utils/node_builders/convolution.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/shape_of.hpp"

namespace ov {
namespace test {
namespace utils {
std::shared_ptr<ov::Model> make_kso_function(ov::Shape input_shape, ov::element::Type type) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(type, input_shape),
                               std::make_shared<ov::op::v0::Parameter>(type, input_shape)};

    auto shape_of = std::make_shared<ov::op::v3::ShapeOf>(params[0]);
    auto convert = std::make_shared<ov::op::v0::Convert>(shape_of, type);
    auto new_shape =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1, 4, 1, 1});
    auto reshape = std::make_shared<ov::op::v1::Reshape>(convert, new_shape, false);

    auto conv1 = ov::test::utils::make_convolution(params[1],
                                                   type,
                                                   {3, 3},
                                                   {1, 1},
                                                   {0, 0},
                                                   {0, 0},
                                                   {1, 1},
                                                   ov::op::PadType::EXPLICIT,
                                                   4);

    auto relu1 = std::make_shared<ov::op::v0::Relu>(conv1);
    auto add = std::make_shared<ov::op::v1::Add>(relu1, reshape);

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(add)};

    std::shared_ptr<ov::Model> model = std::make_shared<ov::Model>(results, params);
    model->set_friendly_name("KSOFunction");
    return model;
}
}  // namespace utils
}  // namespace test
}  // namespace ov