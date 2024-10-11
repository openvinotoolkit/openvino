// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_convolution.hpp"

#include "intel_gpu/op/convolution.hpp"
#include "intel_gpu/op/placeholder.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/util/convolution_base.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

#include <memory>

using namespace ov::pass::pattern;
using ov::pass::pattern::op::Or;

namespace ov {
namespace intel_gpu {
namespace {

template<typename W_T, typename AZP_T>
ov::Tensor get_compensation(ov::Tensor* w_tensor, ov::Tensor* azp_tensor, ov::Tensor* wzp_tensor, int64_t groups) {
    const auto weights_shape = w_tensor->get_shape();
    const auto output_channels = groups > 0 ? weights_shape[1] : weights_shape[0];
    const auto input_channels = groups > 0 ? weights_shape[2] : weights_shape[1];
    size_t azp_total = azp_tensor ? azp_tensor->get_size() : 1;
    size_t wzp_total = wzp_tensor ? wzp_tensor->get_size() : 1;
    const size_t groups_count = std::max<int64_t>(groups, 1);
    const auto total_spatial_size = ov::shape_size(weights_shape) / input_channels / output_channels / groups_count;
    size_t weights_rank = w_tensor->get_shape().size();
    ov::Shape compensation_shape(weights_rank - (groups > 0 ? 1 : 0), 1);
    compensation_shape[1] = output_channels * groups_count;
    ov::Tensor compensation(ov::element::f32, compensation_shape);

    float* comp = compensation.data<float>();
    const W_T* w = w_tensor->data<const W_T>();
    const W_T* wzp = wzp_tensor ? wzp_tensor->data<const W_T>() : nullptr;
    const AZP_T* azp = azp_tensor ? azp_tensor->data<const AZP_T>() : nullptr;

    for (size_t g = 0; g < groups_count; g++) {
        for (size_t oc = 0; oc < output_channels; oc++) {
            float c = 0.f;
            for (size_t ic = 0; ic < input_channels; ic++) {
                for (size_t k = 0; k < total_spatial_size; k++) {
                    size_t azp_offset = (g * input_channels + ic) % azp_total;
                    size_t wzp_offset = (g * output_channels + oc) % wzp_total;
                    const auto w_offset = g * output_channels * input_channels * total_spatial_size
                                        + oc * input_channels * total_spatial_size
                                        + ic * total_spatial_size
                                        + k;

                    if (azp) {
                        c += w[w_offset] * azp[azp_offset];
                        if (wzp) {
                            c -= azp[azp_offset] * wzp[wzp_offset];
                        }
                    }
                }
            }
            comp[(g * output_channels + oc)] = -c;
        }
    }

    return compensation;
}

ov::Tensor get_compensation(std::shared_ptr<ov::Node> w, std::shared_ptr<ov::Node> azp, std::shared_ptr<ov::Node> wzp, int64_t groups) {
    auto w_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(w);
    auto azp_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(azp);
    auto wzp_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(wzp);

    OPENVINO_ASSERT(w_const != nullptr && azp_const != nullptr);

    ov::Tensor w_tensor(w_const->get_element_type(), w_const->get_shape(), const_cast<void*>(w_const->get_data_ptr()));
    ov::Tensor azp_tensor(azp_const->get_element_type(), azp_const->get_shape(), const_cast<void*>(azp_const->get_data_ptr()));
    ov::Tensor wzp_tensor;
    if (wzp_const) {
        wzp_tensor = ov::Tensor(wzp_const->get_element_type(), wzp_const->get_shape(), const_cast<void*>(wzp_const->get_data_ptr()));
    }

    if (w_const->get_element_type() == ov::element::u8 && azp_const->get_element_type() == ov::element::u8)
        return get_compensation<uint8_t, uint8_t>(&w_tensor, &azp_tensor, wzp_const ? &wzp_tensor : nullptr, groups);
    else if (w_const->get_element_type() == ov::element::u8 && azp_const->get_element_type() == ov::element::i8)
        return get_compensation<uint8_t, int8_t>(&w_tensor, &azp_tensor, wzp_const ? &wzp_tensor : nullptr, groups);
    else if (w_const->get_element_type() == ov::element::i8 && azp_const->get_element_type() == ov::element::u8)
        return get_compensation<int8_t, uint8_t>(&w_tensor, &azp_tensor, wzp_const ? &wzp_tensor : nullptr, groups);
    else if (w_const->get_element_type() == ov::element::i8 && azp_const->get_element_type() == ov::element::i8)
        return get_compensation<int8_t, int8_t>(&w_tensor, &azp_tensor, wzp_const ? &wzp_tensor : nullptr, groups);

    OPENVINO_THROW("[GPU] Unsupported element types combination for quantized weights and zero-points");
}

}  // namespace

class ConvolutionMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvolutionMatcher", "0");
    ConvolutionMatcher();
};

class AsymmetricConvolutionMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("AsymmetricConvolutionMatcher", "0");
    AsymmetricConvolutionMatcher();
};

AsymmetricConvolutionMatcher::AsymmetricConvolutionMatcher() {
    auto input_m = any_input(type_matches_any({ov::element::u8, ov::element::i8}));
    auto azp_const_m = wrap_type<ov::op::v0::Constant>(all_of({consumers_count(1), type_matches_any({ov::element::u8, ov::element::i8})}));
    auto azp_subtract_m = wrap_type<ov::op::v1::Subtract>({input_m, azp_const_m});

    auto weights_m = wrap_type<ov::op::v0::Constant>(type_matches_any({ov::element::u8, ov::element::i8}));
    auto wzp_const_m = wrap_type<ov::op::v0::Constant>(all_of({consumers_count(1), type_matches_any({ov::element::u8, ov::element::i8})}));
    auto wzp_subtract_m = wrap_type<ov::op::v1::Subtract>({weights_m, wzp_const_m});

    auto conv_activations_m = std::make_shared<Or>(OutputVector{input_m, azp_subtract_m});
    auto conv_weights_m = std::make_shared<Or>(OutputVector{weights_m, wzp_subtract_m});

    auto convolution_m = wrap_type<ov::op::v1::Convolution, ov::op::v1::GroupConvolution>({ conv_activations_m, conv_weights_m });

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        if (transformation_callback(m.get_match_root())) {
            return false;
        }

        auto conv_node = std::dynamic_pointer_cast<ov::op::util::ConvolutionFwdPropBase>(pattern_map.at(convolution_m).get_node_shared_ptr());

        int64_t groups = -1;
        if (auto grouped_conv = std::dynamic_pointer_cast<ov::op::v1::GroupConvolution>(conv_node)) {
            auto weights_shape = grouped_conv->get_input_partial_shape(1);
            if (weights_shape[0].is_dynamic())
                return false;
            groups = weights_shape[0].get_length();
        }

        auto weights = std::dynamic_pointer_cast<ov::op::v0::Constant>(pattern_map.at(weights_m).get_node_shared_ptr());
        std::shared_ptr<ov::Node> no_bias = std::make_shared<op::Placeholder>();
        std::shared_ptr<ov::Node> optional_wzp_point = std::make_shared<op::Placeholder>();
        std::shared_ptr<ov::Node> optional_azp_point = std::make_shared<op::Placeholder>();
        std::shared_ptr<ov::Node> optional_compensation = std::make_shared<op::Placeholder>();
        auto activations = pattern_map.at(input_m);

        ov::Tensor compensation_tensor;
        const bool with_wzp = pattern_map.count(wzp_subtract_m) > 0;
        if (with_wzp) {
            optional_wzp_point = pattern_map.at(wzp_const_m).get_node_shared_ptr();
        }
        const bool with_azp = pattern_map.count(azp_subtract_m) > 0;
        if (with_azp) {
            optional_azp_point = pattern_map.at(azp_const_m).get_node_shared_ptr();
            compensation_tensor = get_compensation(weights, optional_azp_point, optional_wzp_point, groups);
            optional_compensation = std::make_shared<ov::op::v0::Constant>(compensation_tensor);
        }

        auto new_conv = std::make_shared<op::Convolution>(activations,
                                                          weights,
                                                          no_bias,
                                                          optional_azp_point,
                                                          optional_wzp_point,
                                                          optional_compensation,
                                                          conv_node->get_strides(),
                                                          conv_node->get_pads_begin(),
                                                          conv_node->get_pads_end(),
                                                          conv_node->get_dilations(),
                                                          groups,
                                                          conv_node->get_auto_pad(),
                                                          conv_node->get_output_element_type(0));
        new_conv->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), new_conv);
        ov::replace_node(m.get_match_root(), new_conv);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(convolution_m, "AsymmetricConvolutionMatcher");
    this->register_matcher(m, callback);
}

ConvolutionMatcher::ConvolutionMatcher() {
    auto input_m = any_input();
    auto weights_m = any_input(has_static_dim(0));
    auto bias_val_m = wrap_type<ov::op::v0::Constant>();
    auto convolution_m = wrap_type<ov::op::v1::Convolution, ov::op::v1::GroupConvolution>({ input_m, weights_m });

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        if (transformation_callback(m.get_match_root())) {
            return false;
        }

        auto conv_node = std::dynamic_pointer_cast<ov::op::util::ConvolutionFwdPropBase>(pattern_map.at(convolution_m).get_node_shared_ptr());

        int64_t groups = -1;
        if (auto grouped_conv = std::dynamic_pointer_cast<ov::op::v1::GroupConvolution>(conv_node)) {
            auto weights_shape = grouped_conv->get_input_partial_shape(1);
            if (weights_shape[0].is_dynamic())
                return false;
            groups = weights_shape[0].get_length();
        }

        auto new_conv = std::make_shared<op::Convolution>(pattern_map.at(input_m),
                                                          pattern_map.at(weights_m),
                                                          std::make_shared<op::Placeholder>(),
                                                          conv_node->get_strides(),
                                                          conv_node->get_pads_begin(),
                                                          conv_node->get_pads_end(),
                                                          conv_node->get_dilations(),
                                                          groups,
                                                          conv_node->get_auto_pad(),
                                                          conv_node->get_output_element_type(0));
        new_conv->set_friendly_name(conv_node->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), new_conv);
        ov::replace_node(m.get_match_root(), new_conv);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(convolution_m, "ConvolutionMatcher");
    this->register_matcher(m, callback);
}

bool ConvertConvolutionToInternal::run_on_model(const std::shared_ptr<ov::Model>& m) {
    ov::pass::Manager manager("ConvertConvolutionToInternal");
    auto pass_config = manager.get_pass_config();
    manager.set_per_pass_validation(false);
    manager.register_pass<AsymmetricConvolutionMatcher>();
    manager.register_pass<ConvolutionMatcher>();

    return manager.run_passes(m);
}

}  // namespace intel_gpu
}  // namespace ov
