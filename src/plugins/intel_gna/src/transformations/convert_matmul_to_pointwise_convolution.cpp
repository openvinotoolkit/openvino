// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "transformations/convert_matmul_to_pointwise_convolution.hpp"

#include <ngraph/opsets/opset7.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <openvino/cc/ngraph/itt.hpp>

#include "backend/gna_limitations.hpp"
#include "layers/gna_permute.hpp"
#include "log/log.hpp"

using namespace ov::intel_gna;
using namespace ov::intel_gna::pass;
using namespace ov::intel_gna::limitations;

static bool BiasValidation(const ngraph::Output<ngraph::Node>& output) {
    auto bias_output_shape = output.get_node()->get_output_shape(0);
    if (bias_output_shape.size() > 4) {
        log::debug() << "bias output shape (" << output.get_node()->get_friendly_name() << ") is more than 4\n";
        return false;
    }

    if (bias_output_shape.size() == 1) {
        return true;
    }

    return std::count_if(bias_output_shape.begin(), bias_output_shape.end(), [](size_t el) {
               return el > 1;
           }) < 2;
}

static std::tuple<bool, uint32_t, uint32_t, uint32_t> VerifyAndGetConvParams(
    std::shared_ptr<ngraph::Node> matmul_node) {
    auto input1_shape = matmul_node->get_input_shape(0);
    auto input2_shape = matmul_node->get_input_shape(1);
    auto output_shape = matmul_node->get_output_shape(0);
    if (input1_shape.size() == 3 && input1_shape.front() == 1) {
        input1_shape.erase(std::begin(input1_shape));
    }

    if (input1_shape.size() != 2 || input2_shape.size() != 2 || output_shape.size() < 2) {
        return std::make_tuple(false, 0, 0, 0);
    }

    // Check if MatMul or corresponding pointwise convolution are supported by GNA
    const uint32_t width = static_cast<uint32_t>(input1_shape.front());
    const uint32_t in_channels = static_cast<uint32_t>(input2_shape.back());
    const uint32_t out_channels = static_cast<uint32_t>(input2_shape.front());
    if (input1_shape.front() <= Limitations::kAffineMaxBatchSize ||
        out_channels % Limitations::kConvFiltersNumDivider != 0 || out_channels > Limitations::kConvMaxFiltersNum ||
        in_channels > Limitations::kConvFilterMaxSize) {
        return std::make_tuple(false, 0, 0, 0);
    }

    return std::make_tuple(true, width, in_channels, out_channels);
}

static bool Convert(std::shared_ptr<ngraph::Node> matmul_node,
                    std::shared_ptr<ngraph::Node> add,
                    std::shared_ptr<ngraph::Node> bias,
                    std::shared_ptr<ngraph::Node> fq) {
    bool supported;
    uint32_t width, in_channels, out_channels;
    std::tie(supported, width, in_channels, out_channels) = VerifyAndGetConvParams(matmul_node);
    if (!supported)
        return false;

    auto input_node = matmul_node->input_value(0).get_node_shared_ptr();
    auto weights_node = matmul_node->input_value(1).get_node_shared_ptr();
    auto base_name = matmul_node->get_friendly_name();

    auto reshape_const_before = std::make_shared<ngraph::opset7::Constant>(ngraph::element::Type_t::i64,
                                                                           ngraph::Shape{4},
                                                                           ngraph::Shape{1, 1, width, in_channels});
    auto reshape_before = std::make_shared<ngraph::opset7::Reshape>(input_node, reshape_const_before, false);
    reshape_before->set_friendly_name(base_name + "/reshape_in");
    ngraph::copy_runtime_info(input_node, reshape_before);

    auto transpose_before = std::make_shared<ngraph::opset7::Transpose>(
        reshape_before,
        ngraph::opset7::Constant::create(
            ngraph::element::i64,
            ngraph::Shape{4},
            permute::GetPermuteOrder(InferenceEngine::Layout::NHWC, InferenceEngine::Layout::NCHW)));
    transpose_before->set_friendly_name(base_name + "/transpose_in");
    ngraph::copy_runtime_info(matmul_node, transpose_before);

    auto weights_reshape_const =
        std::make_shared<ngraph::opset7::Constant>(ngraph::element::Type_t::i64,
                                                   ngraph::Shape{4},
                                                   ngraph::Shape{out_channels, in_channels, 1, 1});
    auto weights_reshaped = std::make_shared<ngraph::opset7::Reshape>(weights_node, weights_reshape_const, false);
    ngraph::copy_runtime_info(weights_node, weights_reshaped);

    std::shared_ptr<ngraph::Node> conv_node =
        std::make_shared<ngraph::opset7::Convolution>(transpose_before,
                                                      weights_reshaped,
                                                      ngraph::Strides{1, 1},
                                                      ngraph::CoordinateDiff{0, 0},
                                                      ngraph::CoordinateDiff{0, 0},
                                                      ngraph::Strides{1, 1},
                                                      ngraph::op::PadType::VALID);
    conv_node->set_friendly_name(base_name + "/conv");
    ngraph::copy_runtime_info(transpose_before, conv_node);

    std::shared_ptr<ngraph::Node> root_node = matmul_node;
    if (bias) {
        auto bias_output_shape = bias->get_output_shape(0);
        std::shared_ptr<ngraph::Node> new_bias = bias;
        if (bias_output_shape.size() > 1 || bias_output_shape.at(0) > 1) {
            std::vector<size_t> axes(4, 1);
            auto iter = std::find_if(bias_output_shape.begin(), bias_output_shape.end(), [](size_t value) {
                return value > 1;
            });
            if (iter != bias_output_shape.end()) {
                axes.at(1) = *iter;
            }
            new_bias = std::make_shared<ngraph::opset7::Constant>(
                bias->get_output_element_type(0),
                axes,
                std::dynamic_pointer_cast<ngraph::opset7::Constant>(bias)->get_data_ptr());
        }

        conv_node = std::make_shared<ngraph::opset7::Add>(conv_node, new_bias);
        ngraph::copy_runtime_info(transpose_before, conv_node);
        root_node = add;
    }

    if (fq != nullptr) {
        conv_node = fq->clone_with_new_inputs(
            {conv_node, fq->input_value(1), fq->input_value(2), fq->input_value(3), fq->input_value(4)});
        ngraph::copy_runtime_info(fq, conv_node);
        root_node = fq;
    }

    auto transpose_after = std::make_shared<ngraph::opset7::Transpose>(
        conv_node,
        ngraph::opset7::Constant::create(
            ngraph::element::i64,
            ngraph::Shape{4},
            permute::GetPermuteOrder(InferenceEngine::Layout::NCHW, InferenceEngine::Layout::NHWC)));
    transpose_after->set_friendly_name(base_name + "/transpose_out");
    ngraph::copy_runtime_info(conv_node, transpose_after);

    auto output_shape = matmul_node->get_output_shape(0);
    output_shape[output_shape.size() - 1] = out_channels;
    output_shape[output_shape.size() - 2] = width;
    auto reshape_const_after = std::make_shared<ngraph::opset7::Constant>(ngraph::element::Type_t::i64,
                                                                          ngraph::Shape{output_shape.size()},
                                                                          output_shape);
    auto reshape_after = std::make_shared<ngraph::opset7::Reshape>(transpose_after, reshape_const_after, false);
    reshape_after->set_friendly_name(base_name);
    ngraph::copy_runtime_info(transpose_after, reshape_after);

    ngraph::replace_node(root_node, reshape_after);
    return true;
}

ConvertMatmulToPointWiseConvolution::ConvertMatmulToPointWiseConvolution() {
    MATCHER_SCOPE(ConvertMatmulToPointWiseConvolution);
    auto const_input = ngraph::pattern::wrap_type<ngraph::opset7::Constant>();
    auto const_fq = ngraph::pattern::wrap_type<ngraph::opset7::FakeQuantize>(
        {const_input,
         ngraph::pattern::wrap_type<ngraph::opset7::Constant>(),
         ngraph::pattern::wrap_type<ngraph::opset7::Constant>(),
         ngraph::pattern::wrap_type<ngraph::opset7::Constant>(),
         ngraph::pattern::wrap_type<ngraph::opset7::Constant>()});
    auto second_input = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{const_input, const_fq});
    auto matmul = ngraph::pattern::wrap_type<ngraph::opset7::MatMul>({ngraph::pattern::any_input(), second_input});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        return Convert(pattern_map.at(matmul).get_node_shared_ptr(), nullptr, nullptr, nullptr);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(matmul, matcher_name);
    this->register_matcher(m, callback);
}

ConvertMatmulWithBiasToPointWiseConvolution::ConvertMatmulWithBiasToPointWiseConvolution() {
    MATCHER_SCOPE(ConvertMatmulWithBiasToPointWiseConvolution);
    auto const_input = ngraph::pattern::wrap_type<ngraph::opset7::Constant>();
    auto const_fq = ngraph::pattern::wrap_type<ngraph::opset7::FakeQuantize>(
        {const_input,
         ngraph::pattern::wrap_type<ngraph::opset7::Constant>(),
         ngraph::pattern::wrap_type<ngraph::opset7::Constant>(),
         ngraph::pattern::wrap_type<ngraph::opset7::Constant>(),
         ngraph::pattern::wrap_type<ngraph::opset7::Constant>()});
    auto second_input = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{const_input, const_fq});
    auto matmul = ngraph::pattern::wrap_type<ngraph::opset7::MatMul>({ngraph::pattern::any_input(), second_input});
    auto bias = ngraph::pattern::wrap_type<ngraph::opset7::Constant>(BiasValidation);
    auto add = ngraph::pattern::wrap_type<ngraph::opset7::Add>({matmul, bias});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        return Convert(pattern_map.at(matmul).get_node_shared_ptr(),
                       pattern_map.at(add).get_node_shared_ptr(),
                       pattern_map.at(bias).get_node_shared_ptr(),
                       nullptr);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(add, matcher_name);
    this->register_matcher(m, callback);
}

ConvertMatmulWithFqToPointWiseConvolution::ConvertMatmulWithFqToPointWiseConvolution() {
    MATCHER_SCOPE(ConvertMatmulWithFqToPointWiseConvolution);
    auto const_input = ngraph::pattern::wrap_type<ngraph::opset7::Constant>();
    auto const_fq = ngraph::pattern::wrap_type<ngraph::opset7::FakeQuantize>(
        {const_input,
         ngraph::pattern::wrap_type<ngraph::opset7::Constant>(),
         ngraph::pattern::wrap_type<ngraph::opset7::Constant>(),
         ngraph::pattern::wrap_type<ngraph::opset7::Constant>(),
         ngraph::pattern::wrap_type<ngraph::opset7::Constant>()});
    auto second_input = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{const_input, const_fq});
    auto matmul = ngraph::pattern::wrap_type<ngraph::opset7::MatMul>({ngraph::pattern::any_input(), second_input});
    auto bias = ngraph::pattern::wrap_type<ngraph::opset7::Constant>(BiasValidation);
    auto add = ngraph::pattern::wrap_type<ngraph::opset7::Add>({matmul, bias});
    auto matmul_out = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{add, matmul});
    auto out_fq = ngraph::pattern::wrap_type<ngraph::opset7::FakeQuantize>(
        {matmul_out,
         ngraph::pattern::wrap_type<ngraph::opset7::Constant>(),
         ngraph::pattern::wrap_type<ngraph::opset7::Constant>(),
         ngraph::pattern::wrap_type<ngraph::opset7::Constant>(),
         ngraph::pattern::wrap_type<ngraph::opset7::Constant>()});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto add_it = pattern_map.find(add);
        auto add_node = (add_it == std::end(pattern_map) ? nullptr : add_it->second.get_node_shared_ptr());
        auto bias_it = pattern_map.find(bias);
        auto bias_node = (bias_it == std::end(pattern_map) ? nullptr : bias_it->second.get_node_shared_ptr());
        return Convert(pattern_map.at(matmul).get_node_shared_ptr(),
                       add_node,
                       bias_node,
                       pattern_map.at(out_fq).get_node_shared_ptr());
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(out_fq, matcher_name);
    this->register_matcher(m, callback);
}
