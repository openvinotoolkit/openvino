// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_deconvolution.hpp"

#include "intel_gpu/op/deconvolution.hpp"
#include "intel_gpu/op/placeholder.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/util/convolution_backprop_base.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

#include <memory>

using namespace ov::pass::pattern;
using ov::pass::pattern::op::Or;

namespace ov {
namespace intel_gpu {

class DeconvolutionMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("DeconvolutionMatcher", "0");
    DeconvolutionMatcher();
};

class DeconvolutionWithOutputShapeMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("DeconvolutionWithOutputShapeMatcher", "0");
    DeconvolutionWithOutputShapeMatcher();
};

DeconvolutionWithOutputShapeMatcher::DeconvolutionWithOutputShapeMatcher() {
    auto input_m = any_input();
    auto weights_m = any_input(has_static_dim(0));
    auto output_shape_m = any_input();
    auto deconvolution_m = wrap_type<ov::op::v1::ConvolutionBackpropData, ov::op::v1::GroupConvolutionBackpropData>({ input_m, weights_m, output_shape_m });

    ov::matcher_pass_callback callback = [=](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        if (transformation_callback(m.get_match_root())) {
            return false;
        }

        auto deconv_node = std::dynamic_pointer_cast<ov::op::util::ConvolutionBackPropBase>(pattern_map.at(deconvolution_m).get_node_shared_ptr());

        int64_t groups = -1;
        auto grouped_deconv = std::dynamic_pointer_cast<ov::op::v1::GroupConvolutionBackpropData>(deconv_node);
        if (grouped_deconv) {
            auto weights_shape = grouped_deconv->get_input_partial_shape(1);
            if (weights_shape[0].is_dynamic())
                return false;
            groups = weights_shape[0].get_length();
        }

        std::shared_ptr<ov::Node> bias = std::make_shared<op::Placeholder>();

        auto new_deconv = std::make_shared<op::Deconvolution>(pattern_map.at(input_m),
                                                          pattern_map.at(weights_m),
                                                          bias,
                                                          pattern_map.at(output_shape_m),
                                                          deconv_node->get_strides(),
                                                          deconv_node->get_pads_begin(),
                                                          deconv_node->get_pads_end(),
                                                          deconv_node->get_dilations(),
                                                          groups,
                                                          deconv_node->get_auto_pad(),
                                                          deconv_node->get_output_element_type(0),
                                                          deconv_node->get_output_padding());
        new_deconv->set_friendly_name(deconv_node->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), new_deconv);
        ov::replace_node(m.get_match_root(), new_deconv);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(deconvolution_m, "DeconvolutionWithOutputShapeMatcher");
    this->register_matcher(m, callback);
}


DeconvolutionMatcher::DeconvolutionMatcher() {
    auto input_m = any_input();
    auto weights_m = any_input(has_static_dim(0));
    auto deconvolution_m = wrap_type<ov::op::v1::ConvolutionBackpropData, ov::op::v1::GroupConvolutionBackpropData>({ input_m, weights_m });

    ov::matcher_pass_callback callback = [=](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        if (transformation_callback(m.get_match_root())) {
            return false;
        }

        auto deconv_node = std::dynamic_pointer_cast<ov::op::util::ConvolutionBackPropBase>(pattern_map.at(deconvolution_m).get_node_shared_ptr());

        int64_t groups = -1;
        auto grouped_deconv = std::dynamic_pointer_cast<ov::op::v1::GroupConvolutionBackpropData>(deconv_node);
        if (grouped_deconv) {
            auto weights_shape = grouped_deconv->get_input_partial_shape(1);
            if (weights_shape[0].is_dynamic())
                return false;
            groups = weights_shape[0].get_length();
        }

        std::shared_ptr<ov::Node> bias = std::make_shared<op::Placeholder>();

        auto new_deconv = std::make_shared<op::Deconvolution>(pattern_map.at(input_m),
                                                          pattern_map.at(weights_m),
                                                          bias,
                                                          deconv_node->get_strides(),
                                                          deconv_node->get_pads_begin(),
                                                          deconv_node->get_pads_end(),
                                                          deconv_node->get_dilations(),
                                                          groups,
                                                          deconv_node->get_auto_pad(),
                                                          deconv_node->get_output_element_type(0),
                                                          deconv_node->get_output_padding());
        new_deconv->set_friendly_name(deconv_node->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), new_deconv);
        ov::replace_node(m.get_match_root(), new_deconv);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(deconvolution_m, "DeconvolutionMatcher");
    this->register_matcher(m, callback);
}

bool ConvertDeconvolutionToInternal::run_on_model(const std::shared_ptr<ov::Model>& m) {
    ov::pass::Manager manager;
    auto pass_config = manager.get_pass_config();
    manager.set_per_pass_validation(false);
    manager.register_pass<DeconvolutionMatcher>();
    manager.register_pass<DeconvolutionWithOutputShapeMatcher>();

    return manager.run_passes(m);
}

}  // namespace intel_gpu
}  // namespace ov
