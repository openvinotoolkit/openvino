// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/graph_comparator.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/pass/manager.hpp"
#include "shared_test_classes/subgraph/mul_conv_fusion.hpp"
#include "transformations/common_optimizations/mul_conv_fusion.hpp"

namespace ov {
namespace test {

std::string MulConvFusion::getTestCaseName(const testing::TestParamInfo<MulConvFusionParams>& obj) {
    ov::NodeTypeInfo conv_type;
    ov::Shape input_shape, weights_shape, const_shape;
    ov::element::Type precision;
    std::string device;
    std::tie(conv_type, input_shape, weights_shape, const_shape, precision, std::ignore, device) = obj.param;
    std::ostringstream results;

    results << conv_type.name << "_";
    results << "input" << ov::test::utils::vec2str(input_shape) << "_";
    results << "weights" << ov::test::utils::vec2str(weights_shape) << "_";
    results << "const" << ov::test::utils::vec2str(const_shape) << "_";
    results << "precision=" << precision << "_";
    results << "device=" << device;
    return results.str();
}

void MulConvFusion::SetUp() {
    ov::NodeTypeInfo conv_type;
    ov::Shape input_shape, weights_shape, const_shape;
    ov::element::Type precision;
    bool is_negative;
    std::tie(conv_type, input_shape, weights_shape, const_shape, precision, is_negative, targetDevice) =
        this->GetParam();
    auto param = std::make_shared<ov::op::v0::Parameter>(precision, input_shape);
    auto spatial_dims = input_shape.size() - 2;

    auto mul_const = ov::test::utils::make_constant(precision, const_shape);
    auto mul = std::make_shared<ov::op::v1::Multiply>(param, mul_const);
    ov::Shape strides(spatial_dims, 1);
    std::vector<ptrdiff_t> pad_begin(spatial_dims, 0), pad_end(spatial_dims, 0);
    auto weights = ov::test::utils::make_constant(precision, weights_shape);
    std::shared_ptr<ov::Node> conv;
    if (conv_type == ov::op::v1::Convolution::get_type_info_static()) {
        conv = std::make_shared<ov::op::v1::Convolution>(mul, weights, strides, pad_begin, pad_end, strides);
    } else if (conv_type == ov::op::v1::GroupConvolution::get_type_info_static()) {
        conv = std::make_shared<ov::op::v1::GroupConvolution>(mul, weights, strides, pad_begin, pad_end, strides);
    } else if (conv_type == ov::op::v1::ConvolutionBackpropData::get_type_info_static()) {
        conv =
            std::make_shared<ov::op::v1::ConvolutionBackpropData>(mul, weights, strides, pad_begin, pad_end, strides);
    } else if (conv_type == ov::op::v1::GroupConvolutionBackpropData::get_type_info_static()) {
        conv = std::make_shared<ov::op::v1::GroupConvolutionBackpropData>(mul,
                                                                          weights,
                                                                          strides,
                                                                          pad_begin,
                                                                          pad_end,
                                                                          strides);
    } else {
        OPENVINO_THROW("Unsupported type");
    }

    function = std::make_shared<ov::Model>(ov::OutputVector{conv}, ov::ParameterVector{param});
    auto cloned_function = function->clone();

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::MultiplyConvolutionFusion>();
    manager.register_pass<ov::pass::MultiplyGroupConvolutionFusion>();
    manager.register_pass<ov::pass::MultiplyConvolutionBackpropDataFusion>();
    manager.register_pass<ov::pass::MultiplyGroupConvolutionBackpropDataFusion>();
    manager.run_passes(cloned_function);

    bool functions_equal = false;
    if (!is_negative) {
        auto param = std::make_shared<ov::op::v0::Parameter>(precision, input_shape);
        ov::Shape strides(spatial_dims, 1);
        std::vector<ptrdiff_t> pad_begin(spatial_dims, 0), pad_end(spatial_dims, 0);
        std::shared_ptr<ov::Node> conv;
        if (conv_type == ov::op::v1::Convolution::get_type_info_static()) {
            weights = std::make_shared<ov::op::v1::Multiply>(weights, mul_const);
            weights = ov::util::get_constant_from_source(weights);
            ASSERT_NE(nullptr, weights);
            conv = std::make_shared<ov::op::v1::Convolution>(param, weights, strides, pad_begin, pad_end, strides);
        } else if (conv_type == ov::op::v1::GroupConvolution::get_type_info_static()) {
            const_shape.insert(const_shape.begin(), weights_shape.size() - const_shape.size(), 1);
            auto G = const_shape[2] > 1 ? weights_shape[0] : 1;
            const_shape[0] = G;
            const_shape[2] /= G;
            auto reshape = std::make_shared<ov::op::v1::Reshape>(
                mul_const,
                ov::op::v0::Constant::create(ov::element::u64, ov::Shape{const_shape.size()}, const_shape),
                false);
            weights = std::make_shared<ov::op::v1::Multiply>(weights, reshape);
            weights = ov::util::get_constant_from_source(weights);
            ASSERT_NE(nullptr, weights);
            conv = std::make_shared<ov::op::v1::GroupConvolution>(param, weights, strides, pad_begin, pad_end, strides);
        } else if (conv_type == ov::op::v1::ConvolutionBackpropData::get_type_info_static()) {
            const_shape.insert(const_shape.begin(), weights_shape.size() - const_shape.size(), 1);
            const_shape[0] = const_shape[1];
            const_shape[1] = 1;
            auto reshape = std::make_shared<ov::op::v1::Reshape>(
                mul_const,
                ov::op::v0::Constant::create(ov::element::u64, ov::Shape{const_shape.size()}, const_shape),
                false);
            weights = std::make_shared<ov::op::v1::Multiply>(weights, reshape);
            weights = ov::util::get_constant_from_source(weights);
            ASSERT_NE(nullptr, weights);
            conv = std::make_shared<ov::op::v1::ConvolutionBackpropData>(param,
                                                                         weights,
                                                                         strides,
                                                                         pad_begin,
                                                                         pad_end,
                                                                         strides);
        } else if (conv_type == ov::op::v1::GroupConvolutionBackpropData::get_type_info_static()) {
            const_shape.insert(const_shape.begin(), weights_shape.size() - const_shape.size(), 1);
            auto G = const_shape[2] > 1 ? weights_shape[0] : 1;
            const_shape[0] = G;
            const_shape[1] = const_shape[2] / G;
            const_shape[2] = 1;
            auto reshape = std::make_shared<ov::op::v1::Reshape>(
                mul_const,
                ov::op::v0::Constant::create(ov::element::u64, ov::Shape{const_shape.size()}, const_shape),
                false);
            weights = std::make_shared<ov::op::v1::Multiply>(weights, reshape);
            weights = ov::util::get_constant_from_source(weights);
            ASSERT_NE(nullptr, weights);
            conv = std::make_shared<ov::op::v1::GroupConvolutionBackpropData>(param,
                                                                              weights,
                                                                              strides,
                                                                              pad_begin,
                                                                              pad_end,
                                                                              strides);
        } else {
            OPENVINO_THROW("Unsupported type");
        }
        auto reference_function = std::make_shared<ov::Model>(ov::OutputVector{conv}, ov::ParameterVector{param});
        std::tie(functions_equal, std::ignore) = compare_functions(cloned_function, reference_function, true);
        ASSERT_TRUE(functions_equal);
    } else {
        auto reference_function = function->clone();
        std::tie(functions_equal, std::ignore) = compare_functions(cloned_function, reference_function, true);
        ASSERT_TRUE(functions_equal);
    }
}
}  // namespace test
}  // namespace ov
