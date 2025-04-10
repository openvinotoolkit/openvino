// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/split_conv_concat.hpp"

#include "common_test_utils/data_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/node_builders/convolution.hpp"

namespace ov {
namespace test {

std::string SplitConvConcat::getTestCaseName(const testing::TestParamInfo<ov::test::BasicParams>& obj) {
    ov::element::Type element_type;
    ov::Shape inputShapes;
    std::string targetDevice;
    std::tie(element_type, inputShapes, targetDevice) = obj.param;

    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(inputShapes) << "_";
    result << "ET=" << element_type << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void SplitConvConcat::SetUp() {
    configure_test(this->GetParam());
}

void SplitConvConcatBase::configure_test(const ov::test::BasicParams& param) {
    ov::Shape inputShape;
    ov::element::Type element_type;
    std::tie(element_type, inputShape, targetDevice) = param;

    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(element_type, ov::Shape(inputShape))};

    auto split_axis_op = std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::i64, ov::Shape{}, std::vector<int64_t>{1});
    auto split = std::make_shared<ov::op::v1::Split>(params[0], split_axis_op, 2);

    std::vector<float> filterWeights1;
    std::vector<float> filterWeights2;
    auto conv1 = ov::test::utils::make_convolution(split->output(0),
                                                  element_type,
                                                  {1, 3},
                                                  {1, 1},
                                                  {0, 0},
                                                  {0, 0},
                                                  {1, 1},
                                                  ov::op::PadType::VALID,
                                                  8,
                                                  false,
                                                  filterWeights1);
    auto relu1 = std::make_shared<ov::op::v0::Relu>(conv1);

    auto conv2 = ov::test::utils::make_convolution(split->output(1),
                                                  element_type,
                                                  {1, 3},
                                                  {1, 1},
                                                  {0, 0},
                                                  {0, 0},
                                                  {1, 1},
                                                  ov::op::PadType::VALID,
                                                  8,
                                                  false,
                                                  filterWeights2);
    auto relu2 = std::make_shared<ov::op::v0::Relu>(conv2);
    auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{relu1->output(0), relu2->output(0)}, 1);

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(concat)};
    function = std::make_shared<ov::Model>(results, params, "SplitConvConcat");
}

}  // namespace test
}  // namespace ov
