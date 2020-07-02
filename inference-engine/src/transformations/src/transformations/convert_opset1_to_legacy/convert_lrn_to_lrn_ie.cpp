// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_opset1_to_legacy/convert_lrn_to_lrn_ie.hpp"

#include <memory>
#include <vector>
#include <string>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>

#include <ngraph_ops/lrn_ie.hpp>

ngraph::pass::ConvertLRNToLegacyMatcher::ConvertLRNToLegacyMatcher() {
    ngraph::handler_callback callback = [](const std::shared_ptr<Node>& node) -> bool {
        auto lrn = std::dynamic_pointer_cast<ngraph::opset1::LRN>(node);
        if (!lrn) {
            return false;
        }

        auto axis_const = std::dynamic_pointer_cast<ngraph::opset1::Constant> (lrn->input(1).get_source_output().get_node_shared_ptr());
        if (!axis_const) {
            return false;
        }

        auto axis_value = axis_const->get_vector<int64_t>();
        std::string region;
        if (axis_value.size() == 1 && axis_value[0] == 1) {
            region = "across";
        } else {
            std::vector<bool> norm(lrn->get_shape().size(), false);
            for (auto & axis : axis_value) {
                if (axis < 0 || axis >= norm.size()) {
                    return false;
                }
                norm[axis] = true;
            }

            // Check that axes belongs to spatial dims
            for (size_t i = 2; i < norm.size(); ++i) {
                if (!norm[i]) return false;
            }
            region = "same";
        }

        auto lrn_ie = std::make_shared<ngraph::op::LRN_IE> (lrn->input(0).get_source_output(),
                                                            lrn->get_alpha(),
                                                            lrn->get_beta(),
                                                            lrn->get_bias(),
                                                            lrn->get_nsize(),
                                                            region);

        lrn_ie->set_friendly_name(lrn->get_friendly_name());
        ngraph::copy_runtime_info(lrn, lrn_ie);
        ngraph::replace_node(lrn, lrn_ie);
        return true;
    };

    this->register_matcher(callback);
}
