// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/subgraph_builders/split_multi_conv_concat.hpp"

#include "common_test_utils/node_builders/convolution.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/split.hpp"

namespace ov {
namespace test {
namespace utils {
std::shared_ptr<ov::Model> make_split_multi_conv_concat(ov::Shape input_shape, ov::element::Type type) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(type, input_shape)};
    params.front()->set_friendly_name("Param_1");
    params.front()->get_output_tensor(0).set_names({"input_tensor"});

    auto split_axis_op =
        std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::i64, ov::Shape{}, std::vector<int64_t>{1});
    auto split = std::make_shared<ov::op::v1::Split>(params[0], split_axis_op, 2);

    auto conv1_0 = ov::test::utils::make_convolution(split->output(0),
                                                     type,
                                                     {3, 3},
                                                     {1, 1},
                                                     {0, 0},
                                                     {0, 0},
                                                     {1, 1},
                                                     ov::op::PadType::EXPLICIT,
                                                     5);
    auto relu1_0 = std::make_shared<ov::op::v0::Relu>(conv1_0);

    auto conv1_1 = ov::test::utils::make_convolution(relu1_0,
                                                     type,
                                                     {3, 3},
                                                     {1, 1},
                                                     {0, 0},
                                                     {0, 0},
                                                     {1, 1},
                                                     ov::op::PadType::EXPLICIT,
                                                     5);
    auto relu1_1 = std::make_shared<ov::op::v0::Relu>(conv1_1);

    auto conv1_2 = ov::test::utils::make_convolution(relu1_1,
                                                     type,
                                                     {3, 3},
                                                     {1, 1},
                                                     {0, 0},
                                                     {0, 0},
                                                     {1, 1},
                                                     ov::op::PadType::EXPLICIT,
                                                     5);
    auto relu1_2 = std::make_shared<ov::op::v0::Relu>(conv1_2);

    auto conv1_3 = ov::test::utils::make_convolution(relu1_2,
                                                     type,
                                                     {3, 3},
                                                     {1, 1},
                                                     {0, 0},
                                                     {0, 0},
                                                     {1, 1},
                                                     ov::op::PadType::EXPLICIT,
                                                     5);
    auto relu1_3 = std::make_shared<ov::op::v0::Relu>(conv1_3);

    auto conv1_4 = ov::test::utils::make_convolution(relu1_2,
                                                     type,
                                                     {3, 3},
                                                     {1, 1},
                                                     {0, 0},
                                                     {0, 0},
                                                     {1, 1},
                                                     ov::op::PadType::EXPLICIT,
                                                     5);
    auto relu1_4 = std::make_shared<ov::op::v0::Relu>(conv1_4);

    auto conv2_0 = ov::test::utils::make_convolution(split->output(1),
                                                     type,
                                                     {3, 3},
                                                     {1, 1},
                                                     {0, 0},
                                                     {0, 0},
                                                     {1, 1},
                                                     ov::op::PadType::EXPLICIT,
                                                     5);
    auto relu2_0 = std::make_shared<ov::op::v0::Relu>(conv2_0);

    auto conv2_1 = ov::test::utils::make_convolution(relu2_0,
                                                     type,
                                                     {3, 3},
                                                     {1, 1},
                                                     {0, 0},
                                                     {0, 0},
                                                     {1, 1},
                                                     ov::op::PadType::EXPLICIT,
                                                     5);
    auto relu2_1 = std::make_shared<ov::op::v0::Relu>(conv2_1);

    auto conv2_2 = ov::test::utils::make_convolution(relu2_1,
                                                     type,
                                                     {3, 3},
                                                     {1, 1},
                                                     {0, 0},
                                                     {0, 0},
                                                     {1, 1},
                                                     ov::op::PadType::EXPLICIT,
                                                     5);
    auto relu2_2 = std::make_shared<ov::op::v0::Relu>(conv2_2);

    auto conv2_3 = ov::test::utils::make_convolution(relu2_2,
                                                     type,
                                                     {3, 3},
                                                     {1, 1},
                                                     {0, 0},
                                                     {0, 0},
                                                     {1, 1},
                                                     ov::op::PadType::EXPLICIT,
                                                     5);
    auto relu2_3 = std::make_shared<ov::op::v0::Relu>(conv2_3);

    auto conv2_4 = ov::test::utils::make_convolution(relu2_2,
                                                     type,
                                                     {3, 3},
                                                     {1, 1},
                                                     {0, 0},
                                                     {0, 0},
                                                     {1, 1},
                                                     ov::op::PadType::EXPLICIT,
                                                     5);
    auto relu2_4 = std::make_shared<ov::op::v0::Relu>(conv2_4);

    auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{relu1_4->output(0), relu2_4->output(0)}, 1);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(concat)};

    std::shared_ptr<ov::Model> model = std::make_shared<ov::Model>(results, params);
    model->set_friendly_name("SplitMultiConvConcat");
    return model;
}
}  // namespace utils
}  // namespace test
}  // namespace ov