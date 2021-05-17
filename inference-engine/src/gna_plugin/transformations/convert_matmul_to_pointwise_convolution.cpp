// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_matmul_to_pointwise_convolution.hpp"

#include <ngraph/opsets/opset7.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

#include "layers/gna_permute.hpp"
#include "backend/gna_limitations.hpp"

using namespace GNAPluginNS;

NGRAPH_RTTI_DEFINITION(ConvertMatmulToPointWiseConvolution, "ConvertMatmulToPointWiseConvolution", 0);

static bool IsTransformationSupportedByNetwork(std::shared_ptr<ngraph::Function> f) {
    /* Currently this transformation is supportrd only for networks without convolutions.
     * TODO: allow it for networks with NHWC convolutions
     */
    for (auto& node : f->get_ordered_ops()) {
        if (std::dynamic_pointer_cast<ngraph::opset7::Convolution>(node) != nullptr) {
            return false;
        }
    }
    return true;
}

static std::tuple<bool, uint32_t, uint32_t, uint32_t> VerifyAndGetConvParams(std::shared_ptr<ngraph::Node> matmul_node) {
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
    const uint32_t width = input1_shape.front();
    const uint32_t in_channels = input2_shape.back();
    const uint32_t out_channels = input2_shape.front();
    if (input1_shape.front() <= GNALimitations::affineMaxBatchSize ||
        out_channels % GNALimitations::convFiltersNumDivider != 0 ||
        out_channels > GNALimitations::convMaxFiltersNum ||
        in_channels > GNALimitations::convFilterMaxSize) {
        return std::make_tuple(false, 0, 0, 0);
    }

    return std::make_tuple(true, width, in_channels, out_channels);
}

static std::vector<int64_t> GetConvSplitSizes(uint32_t width, uint32_t in_channels, uint32_t out_channels) {
    uint32_t usedWidth = 0;
    std::vector<int64_t> split_sizes;
    uint32_t width_max_size = GNALimitations::bufferMaxSize / std::max(in_channels, out_channels);
    width_max_size = width_max_size - width_max_size % 64;
    while (usedWidth < width) {
        uint32_t width_part = std::min(width - usedWidth, width_max_size);
        split_sizes.push_back(width_part);
        usedWidth += width_part;
    }
    IE_ASSERT(usedWidth == width);
    return split_sizes;
}

static std::shared_ptr<ngraph::Node> CreatePointwiseConv(ngraph::Output<ngraph::Node> parent_node,
                                                         std::shared_ptr<ngraph::opset7::Constant> weights_node,
                                                         std::shared_ptr<ngraph::opset7::Constant> bias_node,
                                                         std::shared_ptr<ngraph::opset7::FakeQuantize> fq_act,
                                                         std::shared_ptr<ngraph::opset7::FakeQuantize> fq_weights,
                                                         std::shared_ptr<ngraph::opset7::FakeQuantize> fq_out,
                                                         const ngraph::Shape &filter_shape,
                                                         const std::string &base_name) {
    const auto elem_type = weights_node->get_element_type();
    std::shared_ptr<ngraph::Node> filter;
    if (elem_type == ngraph::element::Type_t::f16) {
        filter = std::make_shared<ngraph::opset7::Constant>(elem_type, filter_shape, weights_node->get_vector<ngraph::float16>());
    } else if (elem_type == ngraph::element::Type_t::f32) {
        filter = std::make_shared<ngraph::opset7::Constant>(elem_type, filter_shape, weights_node->get_vector<float>());
    } else {
        THROW_GNA_EXCEPTION << "Unexpected element type " << elem_type << " for weights: "
                            << weights_node->get_friendly_name();
    }

    if (fq_weights) {
        filter = fq_weights->clone_with_new_inputs({filter, fq_weights->input_value(1),
            fq_weights->input_value(2), fq_weights->input_value(3), fq_weights->input_value(4)});
    }

    auto conv_in = parent_node;
    if (fq_act) {
        conv_in = fq_act->clone_with_new_inputs({parent_node, fq_act->input_value(1), fq_act->input_value(2),
                     fq_act->input_value(3), fq_act->input_value(4)});
        conv_in.get_node_shared_ptr()->set_friendly_name(base_name + "/fq_in");
    }

    auto conv_node = std::make_shared<ngraph::opset7::Convolution>(conv_in, filter,
            ngraph::Strides{1, 1}, ngraph::CoordinateDiff{0, 0}, ngraph::CoordinateDiff{0, 0},
            ngraph::Strides{1, 1}, ngraph::op::PadType::VALID);
    conv_node->set_friendly_name(base_name + "/conv");

    std::shared_ptr<ngraph::Node> conv_out = conv_node;
    if (bias_node) {
        conv_out = std::make_shared<ngraph::opset7::Add>(conv_out, bias_node);
        conv_out->set_friendly_name(base_name + "/add");
    }

    if (fq_out) {
        conv_out = fq_out->clone_with_new_inputs({conv_out, fq_out->input_value(1), fq_out->input_value(2),
                   fq_out->input_value(3), fq_out->input_value(4)});
        conv_out->set_friendly_name(base_name + "/fq_out");
    }
    return conv_out;
}

static void RemoveExtraNodes(std::shared_ptr<ngraph::opset7::Add> bias_add,
                             std::shared_ptr<ngraph::opset7::FakeQuantize> fq_act,
                             std::shared_ptr<ngraph::opset7::FakeQuantize> fq_out) {
    if (bias_add) {
        std::shared_ptr<ngraph::Node> input = bias_add->input_value(0).get_node_shared_ptr();
        if (std::dynamic_pointer_cast<ngraph::opset7::Constant>(input)) {
            input = bias_add->input_value(1).get_node_shared_ptr();
        }
        ngraph::replace_output_update_name(bias_add->output(0), input);
    }

    if (fq_act) {
        ngraph::replace_output_update_name(fq_act->output(0), fq_act->input_value(0));
    }

    if (fq_out) {
        ngraph::replace_output_update_name(fq_out->output(0), fq_out->input_value(0));
    }
}

bool ConvertMatmulToPointWiseConvolution::run_on_function(std::shared_ptr<ngraph::Function> f) {
    if (!IsTransformationSupportedByNetwork(f)) return false;

    bool is_graph_modfied = false;
    for (auto& node : f->get_ordered_ops()) {
        if (std::dynamic_pointer_cast<ngraph::opset7::MatMul>(node) == nullptr) {
            continue;
        }

        bool supported;
        uint32_t width, in_channels, out_channels;
        std::tie(supported, width, in_channels, out_channels) = VerifyAndGetConvParams(node);
        if (!supported) continue;

        auto get_input_skip_fq = [node](size_t input_idx) {
            std::shared_ptr<ngraph::Node> input_skip_fq = node->input_value(input_idx).get_node_shared_ptr();
            auto fq = std::dynamic_pointer_cast<ngraph::opset7::FakeQuantize>(input_skip_fq);
            if (fq) {
                input_skip_fq = input_skip_fq->input_value(0).get_node_shared_ptr();
            }
            return std::make_pair(input_skip_fq, fq);
        };

        std::shared_ptr<ngraph::Node> input_skip_fq, weights_skip_fq;
        std::shared_ptr<ngraph::opset7::FakeQuantize> fq_act, fq_weights;
        std::tie(input_skip_fq, fq_act) = get_input_skip_fq(0);
        std::tie(weights_skip_fq, fq_weights) = get_input_skip_fq(1);
        auto weights_node = std::dynamic_pointer_cast<ngraph::opset7::Constant>(weights_skip_fq);
        if (std::dynamic_pointer_cast<ngraph::opset7::Constant>(input_skip_fq) || !weights_node) {
            continue;
        }

        // If Fake Quantize goes after Matmul or Eltwise operation don't move it since it should be fused with them
        bool move_act_fq = fq_act &&
            !std::dynamic_pointer_cast<ngraph::opset7::MatMul>(fq_act->input_value(0).get_node_shared_ptr()) &&
            !std::dynamic_pointer_cast<ngraph::op::util::BinaryElementwiseArithmetic>(fq_act->input_value(0).get_node_shared_ptr());

        /* Determine if there is a bias and/or fake quantize after matmul, they should go after each point-wise convolution
         * after this transformation
         */
        std::shared_ptr<ngraph::opset7::Constant> bias_node = nullptr;
        auto matmul_output = node->output(0).get_target_inputs().empty() ? nullptr :
            node->output(0).get_target_inputs().begin()->get_node()->shared_from_this();
        auto add = std::dynamic_pointer_cast<ngraph::opset7::Add>(matmul_output);
        if (add) {
            bias_node = std::dynamic_pointer_cast<ngraph::opset7::Constant>(add->input_value(0).get_node_shared_ptr());
            if (!bias_node) {
                bias_node = std::dynamic_pointer_cast<ngraph::opset7::Constant>(add->input_value(1).get_node_shared_ptr());
            }
        }

        std::shared_ptr<ngraph::Node> fq_out_node = nullptr;
        if (bias_node) {
            if (!add->output(0).get_target_inputs().empty()) {
                fq_out_node = add->output(0).get_target_inputs().begin()->get_node()->shared_from_this();
            }
        } else {
            fq_out_node = matmul_output;
        }
        auto fq_out = std::dynamic_pointer_cast<ngraph::opset7::FakeQuantize>(fq_out_node);

        auto reshape_const_before = std::make_shared<ngraph::opset7::Constant>(ngraph::element::Type_t::i64,
                                                                               ngraph::Shape{4},
                                                                               ngraph::Shape{1, 1, width, in_channels});
        auto start_node = (move_act_fq || !fq_act) ? input_skip_fq : fq_act;
        auto reshape_before =  std::make_shared<ngraph::opset7::Reshape>(start_node, reshape_const_before, false);
        reshape_before->set_friendly_name(node->get_friendly_name() + "/reshape_in");

        auto transpose_before = std::make_shared<ngraph::opset7::Transpose>(reshape_before,
            ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{4},
            GetPermuteOrder(InferenceEngine::Layout::NHWC, InferenceEngine::Layout::NCHW)));
        transpose_before->set_friendly_name(node->get_friendly_name() + "/transpose_in");

        auto split_sizes = GetConvSplitSizes(width, in_channels, out_channels);
        std::shared_ptr<ngraph::Node> output;
        const ngraph::Shape filter_shape = {out_channels, in_channels, 1, 1};
        if (split_sizes.size() == 1) {
            output = CreatePointwiseConv(transpose_before, weights_node, bias_node, move_act_fq ? fq_act : nullptr,
                fq_weights, fq_out, filter_shape, node->get_friendly_name());
        } else {
            const size_t width_axis = 3;
            auto split_node = std::make_shared<ngraph::opset7::VariadicSplit>(transpose_before,
                ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape({1}), std::vector<int64_t>{width_axis}),
                ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape({split_sizes.size()}), split_sizes));
            split_node->set_friendly_name(node->get_friendly_name() + "/split");
            ngraph::OutputVector convOutputs;
            for (int i = 0; i < split_sizes.size(); ++i) {
                auto output = CreatePointwiseConv(split_node->output(i), weights_node, bias_node, move_act_fq ? fq_act : nullptr,
                    fq_weights, fq_out, filter_shape, node->get_friendly_name() + "_" + std::to_string(i));
                convOutputs.push_back(output);
            }
            output = std::make_shared<ngraph::opset7::Concat>(convOutputs, width_axis);
            output->set_friendly_name(node->get_friendly_name() + "/concat");
        }

        auto transpose_after = std::make_shared<ngraph::opset7::Transpose>(output,
            ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{4},
            GetPermuteOrder(InferenceEngine::Layout::NCHW, InferenceEngine::Layout::NHWC)));
        transpose_after->set_friendly_name(node->get_friendly_name() + "/transpose_out");

        auto output_shape = node->get_output_shape(0);
        output_shape[output_shape.size() - 1] = out_channels;
        output_shape[output_shape.size() - 2] = width;
        auto reshape_const_after = std::make_shared<ngraph::opset7::Constant>(ngraph::element::Type_t::i64,
                                                                              ngraph::Shape{output_shape.size()},
                                                                              output_shape);
        auto reshape_after =  std::make_shared<ngraph::opset7::Reshape>(transpose_after, reshape_const_after, false);
        reshape_after->set_friendly_name(node->get_friendly_name());

        ngraph::replace_node(node, reshape_after);
        RemoveExtraNodes(bias_node ? add : nullptr, move_act_fq ? fq_act : nullptr, fq_out);
        is_graph_modfied = true;
    }
    return is_graph_modfied;
}