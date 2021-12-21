// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pruning.hpp"
#include "mask_attribute.hpp"

#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/opsets/opset6.hpp>
#include <ngraph/log.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::InitMasks, "InitMasks", 0);

namespace ngraph {
namespace pass {
namespace init_masks {

class InitConvMask;

} // namespace init_masks
} // namespace pass
} // namespace ngraph

class ngraph::pass::init_masks::InitConvMask : public MatcherPass {
public:
    InitConvMask() {
        auto input = pattern::any_input();
        auto weights = pattern::any_input();
        auto conv = pattern::wrap_type<opset6::Convolution, opset6::GroupConvolution>({input, weights});

        ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
            const auto & pattern_map = m.get_pattern_value_map();
            const auto & m_output = pattern_map.at(conv);

            // Initializing weights mask:
            // 1. Looking for Const node with weights
            NodeVector weights_calculation_nodes;
            auto cur_node = m_output.get_node()->get_input_node_shared_ptr(1);

            while (!ngraph::is_type<opset6::Constant>(cur_node) && cur_node->inputs().size()) {
                weights_calculation_nodes.push_back(cur_node);
                cur_node = cur_node->get_input_node_shared_ptr(0);
            }
            if (!ngraph::is_type<opset6::Constant>(cur_node)) {
                NGRAPH_DEBUG << "Can't find Constant weights for Convolution: " <<
                m_output.get_node()->get_friendly_name() << std::endl;
                return false;
            }

            // 2. Init mask for Const node
            InitConstMask({0}/* check only output channels dim */).apply(cur_node);
            return true;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(conv, "ConvolutionInitMask");
        register_matcher(m, callback);
    }
};


ngraph::pass::InitMasks::InitMasks() {
    add_matcher<init_masks::InitConvMask>();
}

