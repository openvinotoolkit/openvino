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

DeconvolutionMatcher::DeconvolutionMatcher() {
    auto input_m = any_input();
    auto weights_m = any_input(has_static_dim(0));
    auto output_shape_m = any_input();
    auto bias_val_m = wrap_type<ov::op::v0::Constant>();
    auto deconvolution_m = wrap_type<ov::op::v1::ConvolutionBackpropData, ov::op::v1::GroupConvolutionBackpropData>({ input_m, weights_m });

    ov::matcher_pass_callback callback = [=](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        if (transformation_callback(m.get_match_root())) {
            return false;
        }

        auto conv_node = std::dynamic_pointer_cast<ov::op::util::ConvolutionBackPropBase>(pattern_map.at(deconvolution_m).get_node_shared_ptr());

        int64_t groups = -1;
        auto grouped_conv = std::dynamic_pointer_cast<ov::op::v1::GroupConvolutionBackpropData>(conv_node);
        if (grouped_conv) {
            auto weights_shape = grouped_conv->get_input_partial_shape(1);
            if (weights_shape[0].is_dynamic())
                return false;
            groups = weights_shape[0].get_length();
        }

        auto new_conv = std::make_shared<op::Deconvolution>(pattern_map.at(input_m),
                                                          pattern_map.at(weights_m),
                                                          conv_node->get_strides(),
                                                          conv_node->get_pads_begin(),
                                                          conv_node->get_pads_end(),
                                                          conv_node->get_dilations(),
                                                          groups,
                                                          conv_node->get_auto_pad(),
                                                          conv_node->get_output_element_type(0),
                                                          conv_node->get_output_padding());
        new_conv->set_friendly_name(conv_node->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), new_conv);
        ov::replace_node(m.get_match_root(), new_conv);
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

    return manager.run_passes(m);
}

}  // namespace intel_gpu
}  // namespace ov
