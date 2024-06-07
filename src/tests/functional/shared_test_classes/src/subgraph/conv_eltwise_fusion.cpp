// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/conv_eltwise_fusion.hpp"

#include "common_test_utils/graph_comparator.hpp"
#include "openvino/core/node.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "transformations/common_optimizations/conv_mul_fusion.hpp"

namespace ov {
namespace test {

std::string ConvEltwiseFusion::getTestCaseName(const testing::TestParamInfo<ConvEltwiseFusionParams>& obj) {
    std::tuple<NodeTypeInfo, size_t> conv_params;
    NodeTypeInfo conv_type, eltwise_type;
    bool negative;
    Shape input_shape, weights_shape, const_shape;
    element::Type precision;
    std::string targetName;
    std::tie(conv_params, eltwise_type, negative, input_shape, weights_shape, const_shape, precision, targetName) =
        obj.param;
    size_t num_inputs;
    std::tie(conv_type, num_inputs) = conv_params;
    std::ostringstream results;

    results << conv_type.name << "_";
    results << "NumInputs=" << num_inputs << "_";
    results << "Negative=" << std::boolalpha << negative << "_";
    results << eltwise_type.name << "_";
    results << "Input" << ov::test::utils::vec2str(input_shape);
    results << "Weights" << ov::test::utils::vec2str(weights_shape);
    results << "Const" << ov::test::utils::vec2str(const_shape);
    results << "netPRC=" << precision << "_";
    results << "targetDevice=" << targetName << "_";
    return results.str();
}

void ConvEltwiseFusion::SetUp() {
    std::tuple<NodeTypeInfo, size_t> conv_params;
    NodeTypeInfo conv_type, eltwise_type;
    bool negative;
    Shape input_shape, weights_shape, const_shape;
    element::Type precision;
    size_t num_inputs;
    std::tie(conv_params, eltwise_type, negative, input_shape, weights_shape, const_shape, precision, targetDevice) =
        this->GetParam();
    std::tie(conv_type, num_inputs) = conv_params;
    pass::Manager manager;

    {
        auto param = std::make_shared<ov::op::v0::Parameter>(precision, input_shape);
        auto spatial_dims = input_shape.size() - 2;

        Shape strides(spatial_dims, 1);
        std::vector<ptrdiff_t> pad_begin(spatial_dims, 0), pad_end(spatial_dims, 0);
        auto weights = ov::op::v0::Constant::create(precision, weights_shape, std::vector<float>(shape_size(weights_shape), 2));
        auto eltwise_const = ov::op::v0::Constant::create(precision, const_shape, std::vector<float>(shape_size(const_shape), 3));
        std::shared_ptr<Node> conv;
        if (conv_type == ov::op::v1::Convolution::get_type_info_static()) {
            conv = std::make_shared<ov::op::v1::Convolution>(param, weights, strides, pad_begin, pad_end, strides);
        } else if (conv_type == ov::op::v1::GroupConvolution::get_type_info_static()) {
            conv = std::make_shared<ov::op::v1::GroupConvolution>(param, weights, strides, pad_begin, pad_end, strides);
        } else if (conv_type == ov::op::v1::ConvolutionBackpropData::get_type_info_static()) {
            if (num_inputs == 3) {
                auto output_shape = std::make_shared<ov::op::v0::Constant>(
                    element::u64,
                    Shape{spatial_dims},
                    std::vector<size_t>(input_shape.begin() + 2, input_shape.end()));
                conv = std::make_shared<ov::op::v1::ConvolutionBackpropData>(param,
                                                                          weights,
                                                                          output_shape,
                                                                          strides,
                                                                          pad_begin,
                                                                          pad_end,
                                                                          strides);
            } else {
                conv = std::make_shared<ov::op::v1::ConvolutionBackpropData>(param,
                                                                          weights,
                                                                          strides,
                                                                          pad_begin,
                                                                          pad_end,
                                                                          strides);
            }
        } else if (conv_type == ov::op::v1::GroupConvolutionBackpropData::get_type_info_static()) {
            if (num_inputs == 3) {
                auto output_shape = std::make_shared<ov::op::v0::Constant>(
                    element::u64,
                    Shape{spatial_dims},
                    std::vector<size_t>(input_shape.begin() + 2, input_shape.end()));
                conv = std::make_shared<ov::op::v1::GroupConvolutionBackpropData>(param,
                                                                               weights,
                                                                               output_shape,
                                                                               strides,
                                                                               pad_begin,
                                                                               pad_end,
                                                                               strides);
            } else {
                conv = std::make_shared<ov::op::v1::GroupConvolutionBackpropData>(param,
                                                                               weights,
                                                                               strides,
                                                                               pad_begin,
                                                                               pad_end,
                                                                               strides);
            }
        } else {
            OPENVINO_THROW("Unsupported type");
        }

        std::shared_ptr<Node> eltwise;
        if (eltwise_type == ov::op::v1::Multiply::get_type_info_static()) {
            eltwise = std::make_shared<ov::op::v1::Multiply>(conv, eltwise_const);
            manager.register_pass<ov::pass::ConvolutionMultiplyFusion>();
            manager.register_pass<ov::pass::GroupConvolutionMultiplyFusion>();
            manager.register_pass<ov::pass::ConvolutionBackpropDataMultiplyFusion>();
            manager.register_pass<ov::pass::GroupConvolutionBackpropDataMultiplyFusion>();
        } else if (eltwise_type == ov::op::v1::Add::get_type_info_static()) {
            eltwise = std::make_shared<ov::op::v1::Add>(conv, eltwise_const);
            // manager.register_pass<pass::ConvertConvolutions>();
            // manager.register_pass<pass::ConvFusion>();
        } else {
            OPENVINO_THROW("Unsupported type");
        }

        function = std::make_shared<Model>(eltwise, ParameterVector{param}, "conv_eltwise");
    }

    manager.register_pass<pass::ConstantFolding>();

    std::shared_ptr<Model> function_ref;

    if (!negative) {
        auto param = std::make_shared<ov::op::v0::Parameter>(precision, input_shape);
        auto spatial_dims = input_shape.size() - 2;

        Shape strides(spatial_dims, 1);
        std::vector<ptrdiff_t> pad_begin(spatial_dims, 0), pad_end(spatial_dims, 0);
        auto weights = ov::op::v0::Constant::create(precision, weights_shape, std::vector<float>(shape_size(weights_shape), 6));
        std::shared_ptr<Node> conv;
        if (conv_type == ov::op::v1::Convolution::get_type_info_static()) {
            conv = std::make_shared<ov::op::v1::Convolution>(param, weights, strides, pad_begin, pad_end, strides);
        } else if (conv_type == ov::op::v1::GroupConvolution::get_type_info_static()) {
            conv = std::make_shared<ov::op::v1::GroupConvolution>(param, weights, strides, pad_begin, pad_end, strides);
        } else if (conv_type == ov::op::v1::ConvolutionBackpropData::get_type_info_static()) {
            if (num_inputs == 3) {
                auto output_shape = std::make_shared<ov::op::v0::Constant>(
                    element::u64,
                    Shape{spatial_dims},
                    std::vector<size_t>(input_shape.begin() + 2, input_shape.end()));
                conv = std::make_shared<ov::op::v1::ConvolutionBackpropData>(param,
                                                                          weights,
                                                                          output_shape,
                                                                          strides,
                                                                          pad_begin,
                                                                          pad_end,
                                                                          strides);
            } else {
                conv = std::make_shared<ov::op::v1::ConvolutionBackpropData>(param,
                                                                          weights,
                                                                          strides,
                                                                          pad_begin,
                                                                          pad_end,
                                                                          strides);
            }
        } else if (conv_type == ov::op::v1::GroupConvolutionBackpropData::get_type_info_static()) {
            if (num_inputs == 3) {
                auto output_shape = std::make_shared<ov::op::v0::Constant>(
                    element::u64,
                    Shape{spatial_dims},
                    std::vector<size_t>(input_shape.begin() + 2, input_shape.end()));
                conv = std::make_shared<ov::op::v1::GroupConvolutionBackpropData>(param,
                                                                               weights,
                                                                               output_shape,
                                                                               strides,
                                                                               pad_begin,
                                                                               pad_end,
                                                                               strides);
            } else {
                conv = std::make_shared<ov::op::v1::GroupConvolutionBackpropData>(param,
                                                                               weights,
                                                                               strides,
                                                                               pad_begin,
                                                                               pad_end,
                                                                               strides);
            }
        }

        function_ref = std::make_shared<Model>(conv, ParameterVector{param}, "conv_eltwise_ref");
    } else {
        function_ref = function->clone();
    }

    auto cloned_function = function->clone();
    manager.run_passes(cloned_function);

    auto res = compare_functions(cloned_function, function_ref);
    ASSERT_TRUE(res.first) << res.second;
}
}  // namespace test
}  // namespace ov
