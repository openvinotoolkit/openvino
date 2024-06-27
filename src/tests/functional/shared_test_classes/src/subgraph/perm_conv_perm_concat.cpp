// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/perm_conv_perm_concat.hpp"

#include "common_test_utils/data_utils.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "common_test_utils/node_builders/convolution.hpp"
#include "common_test_utils/node_builders/constant.hpp"

namespace ov {
namespace test {

std::string PermConvPermConcat::getTestCaseName(const testing::TestParamInfo<PermConvPermConcatParams>& obj) {
    ov::element::Type element_type;
    std::string targetName;
    ov::Shape input_shape;
    ov::Shape kernel_shape;
    size_t output_channels;
    ov::AnyMap configuration;

    std::tie(element_type, targetName, input_shape, kernel_shape, output_channels, configuration) = obj.param;
    std::ostringstream results;

    results << "IS=" << ov::test::utils::vec2str(std::vector<size_t>(input_shape.begin(), input_shape.end())) << "_";
    results << "KS=" << ov::test::utils::vec2str(std::vector<size_t>(kernel_shape.begin(), kernel_shape.end())) << "_";
    results << "OC=" << output_channels << "_";
    results << "ET=" << element_type << "_";
    results << "targetDevice=" << targetName;
    for (auto const& configItem : configuration) {
        results << "_configItem=" << configItem.first << "_" << configItem.second.as<std::string>();
    }
    return results.str();
}

void PermConvPermConcat::SetUp() {
    ov::element::Type element_type;
    ov::Shape input_shape;
    ov::Shape kernel_shape;
    size_t output_channels;
    ov::AnyMap additional_config;

    std::tie(element_type, targetDevice, input_shape, kernel_shape, output_channels, additional_config) =
        this->GetParam();

    if (element_type == ov::element::f32) {
        abs_threshold = 1e-6;
    }

    configuration.insert(additional_config.begin(), additional_config.end());

    const std::size_t input_dim = std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<size_t>());

    std::vector<size_t> input_dims{1, input_dim};
    std::vector<size_t> reshape_in_dims = std::vector<size_t>(input_shape.begin(), input_shape.end());
    std::vector<size_t> permute_in_order = {0, 3, 1, 2};
    std::vector<size_t> permute_out_order = {0, 2, 3, 1};

    ov::ParameterVector input_parameter{std::make_shared<ov::op::v0::Parameter>(element_type, ov::Shape(input_dims))};

    auto reshape_in_pattern = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, reshape_in_dims);
    auto reshape_in = std::make_shared<ov::op::v1::Reshape>(input_parameter[0], reshape_in_pattern, false);

    auto permute_in_params =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, ov::Shape{permute_in_order});
    auto permute_in = std::make_shared<ov::op::v1::Transpose>(reshape_in, permute_in_params);
    auto conv_in_shape = permute_in->get_output_shape(0);
    auto conv_weights_size = output_channels * (conv_in_shape[1]) * kernel_shape[0] * kernel_shape[1];
    auto conv =
        ov::test::utils::make_convolution(permute_in,
                                         element_type,
                                         {kernel_shape[0], kernel_shape[1]},
                                         {1, 1},
                                         {0, 0},
                                         {0, 0},
                                         {1, 1},
                                         ov::op::PadType::VALID,
                                         output_channels,
                                         false,
                                         ov::test::utils::generate_float_numbers(conv_weights_size, -0.5f, 0.5f));

    auto permute_out_params = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, permute_out_order);
    auto permute_out = std::make_shared<ov::op::v1::Transpose>(conv, permute_out_params);

    auto permute_out_shape = permute_out->get_output_shape(0);

    auto concat_const =
        ov::op::v0::Constant::create(element_type, {1, 1, 1, permute_out_shape[3]},
            ov::test::utils::generate_float_numbers(permute_out_shape[3], -10, 10));

    auto concat = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{permute_out, concat_const}, 2);

    auto reshape_out_pattern = std::make_shared<ov::op::v0::Constant>(
        ov::element::i64,
        ov::Shape{2},
        std::vector<size_t>({1, (permute_out_shape[2] + 1) * permute_out_shape[3]}));
    auto reshape_out = std::make_shared<ov::op::v1::Reshape>(concat, reshape_out_pattern, false);

    function = std::make_shared<ov::Model>(reshape_out, input_parameter, "perm_conv_perm_concat");
}

}  // namespace test
}  // namespace ov
