// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "remarks.hpp"
#include "itt.hpp"

#include "snippets/pass/load_movebroadcast_to_broadcastload.hpp"
#include "snippets/snippets_isa.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

ngraph::snippets::pass::LoadMoveBroadcastToBroadcastLoad::LoadMoveBroadcastToBroadcastLoad() {
    MATCHER_SCOPE(LoadMoveBroadcastToBroadcastLoad);
    auto param_pattern = ngraph::pattern::wrap_type<ngraph::opset1::Parameter>();
    auto load_pattern = std::make_shared<ngraph::snippets::op::Load>(param_pattern);
    auto fbn = std::make_shared<ngraph::snippets::op::BroadcastMove>(load_pattern, Shape{1});

    register_matcher(std::make_shared<ngraph::pattern::Matcher>(fbn),
        [load_pattern, param_pattern](ngraph::pattern::Matcher &m) {
            auto root = m.get_match_root();

            const auto &pm = m.get_pattern_value_map();
            const auto input = pm.at(load_pattern).get_node_shared_ptr();
            const auto param = pm.at(param_pattern).get_node_shared_ptr();

            // check if load has more than 1 user to avoid load+broadcast load on the same parameter
            if (input->output(0).get_target_inputs().size() != 1) {
                return false;
            }

            if (root->inputs().size() != 1 || input->inputs().size() != 1) {
                throw ngraph_error("cannot rewrite Broadcast load with more than one input");
            }

            auto inshape = root->input(0).get_shape();
            auto outshape = root->output(0).get_shape();
            auto broadcastload = std::make_shared<snippets::op::BroadcastLoad>(param, outshape);
            Shape bct(inshape.size(), 0);
            for (size_t k = 0; k < inshape.size(); k++) {
                if (inshape[k] != outshape[k] && inshape[k] == 1) {
                    bct[k] = 1;
                }
            }

            broadcastload->set_broadcast_info(bct);
            if (broadcastload->is_broadcast(outshape.size()-1)) {
                ngraph::copy_runtime_info(root, broadcastload);
                ngraph::replace_node(root, broadcastload);
                return true;
            } else {
                return false;
            }
        });
}