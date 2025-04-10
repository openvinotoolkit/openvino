// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/reshape_permute_conv_permute_reshape_act.hpp"
#include "common_test_utils/node_builders/convolution.hpp"

namespace ov {
namespace test {
std::string ConvReshapeAct::getTestCaseName(const testing::TestParamInfo<ConvReshapeActParams>& obj) {
    ov::element::Type model_type;
    std::string targetName;
    std::array<size_t, 4> input_shape;
    std::array<size_t, 2> kernel_shape;
    size_t output_channels;
    ov::AnyMap configuration;


    std::tie(model_type, targetName, input_shape, kernel_shape, output_channels, configuration) = obj.param;
    std::ostringstream results;

    results << "IS=" << ov::test::utils::vec2str(std::vector<size_t>(input_shape.begin(), input_shape.end())) << "_";
    results << "KS=" << ov::test::utils::vec2str(std::vector<size_t>(kernel_shape.begin(), kernel_shape.end())) << "_";
    results << "OC=" << output_channels << "_";
    results << "netPRC=" << model_type.get_type_name() << "_";
    results << "targetDevice=" << targetName;
    for (auto const& configItem : configuration) {
        results << "_configItem=" << configItem.first << "_" << configItem.second.as<std::string>();
    }
    return results.str();
}

void ConvReshapeAct::SetUp() {
    ov::element::Type model_type;
    std::array<size_t, 4> input_shape;
    std::array<size_t, 2> kernel_shape;
    size_t output_channels;
    ov::AnyMap additional_config;

    std::tie(model_type, targetDevice, input_shape, kernel_shape, output_channels, additional_config) = this->GetParam();

    configuration.insert(additional_config.begin(), additional_config.end());

    const std::size_t input_dim = std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<size_t>());

    std::vector<size_t> input_dims { 1, input_dim };
    std::vector<size_t> reshape_in_dims = std::vector<size_t>(input_shape.begin(), input_shape.end());
    std::vector<size_t> permute_in_order = { 0, 3, 1, 2 };
    std::vector<size_t> permute_out_order = { 0, 2, 3, 1 };
    std::vector<size_t> reshape_out_dims = { 1, input_shape[0] * input_shape[1] * (input_shape[2] - kernel_shape[1] + 1) * output_channels };

    ov::ParameterVector input_parameter {std::make_shared<ov::op::v0::Parameter>(model_type, ov::Shape(input_dims))};

    auto reshape_in_pattern = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
        ov::Shape{4},
        reshape_in_dims);
    auto reshape_in = std::make_shared<ov::op::v1::Reshape>(input_parameter[0], reshape_in_pattern, false);

    auto permute_in_params = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
        ov::Shape{4},
        ov::Shape{permute_in_order});
    auto permute_in = std::make_shared<ov::op::v1::Transpose>(reshape_in, permute_in_params);

    auto conv = ov::test::utils::make_convolution(permute_in, model_type, {kernel_shape[0], kernel_shape[1]}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
        ov::op::PadType::VALID, output_channels);

    auto permute_out_params = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
        ov::Shape{4},
        permute_out_order);
    auto permute_out = std::make_shared<ov::op::v1::Transpose>(conv, permute_out_params);

    auto reshape_out_pattern = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
        ov::Shape{2},
        std::vector<size_t>{reshape_out_dims});

    auto reshape_out = std::make_shared<ov::op::v1::Reshape>(permute_out, reshape_out_pattern, false);

    auto tanh = std::make_shared<ov::op::v0::Tanh>(reshape_out);

    function = std::make_shared<ov::Model>(tanh, input_parameter, "conv_reshape_act");
}

} // namespace test
} // namespace ov
