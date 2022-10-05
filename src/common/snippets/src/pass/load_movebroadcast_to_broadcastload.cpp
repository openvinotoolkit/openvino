// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/remarks.hpp"
#include <snippets/itt.hpp>

#include "snippets/pass/load_movebroadcast_to_broadcastload.hpp"
#include "snippets/snippets_isa.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

ngraph::snippets::pass::LoadMoveBroadcastToBroadcastLoad::LoadMoveBroadcastToBroadcastLoad() {
    MATCHER_SCOPE(LoadMoveBroadcastToBroadcastLoad);
    auto param_pattern = ngraph::pattern::wrap_type<ngraph::opset1::Parameter>();
    auto load_pattern = ngraph::pattern::wrap_type<ngraph::snippets::op::Load>({param_pattern});
    auto fbn = std::make_shared<ngraph::snippets::op::BroadcastMove>(load_pattern, Shape{1});

    register_matcher(std::make_shared<ngraph::pattern::Matcher>(fbn, matcher_name),
        [load_pattern, param_pattern](ngraph::pattern::Matcher &m) {
            OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::op::LoadMoveBroadcastToBroadcastLoad")
            auto root = m.get_match_root();

            const auto &pm = m.get_pattern_value_map();
            const auto input = pm.at(load_pattern).get_node_shared_ptr();
            const auto param = pm.at(param_pattern).get_node_shared_ptr();

            // Cannot rewrite Broadcast + Load if load has more than 1 user
            // or more than one input, or if Broadcast has several inputs
            if (input->output(0).get_target_inputs().size() != 1 ||
                root->inputs().size() != 1 || input->inputs().size() != 1) {
                return false;
            }

            auto inshape = root->input(0).get_shape();
            auto outshape = root->output(0).get_shape();

            auto broadcastload = std::make_shared<snippets::op::BroadcastLoad>(param, outshape);
            ngraph::copy_runtime_info(root, broadcastload);
            ngraph::replace_node(root, broadcastload);

            return true;
        });
}