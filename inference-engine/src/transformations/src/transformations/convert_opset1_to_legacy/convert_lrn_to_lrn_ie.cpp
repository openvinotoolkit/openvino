// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_opset1_to_legacy/convert_lrn_to_lrn_ie.hpp"

#include <memory>
#include <vector>
#include <string>

#include <ngraph/opsets/opset1.hpp>

#include <ngraph_ops/lrn_ie.hpp>

void ngraph::pass::ConvertLRNToLRNIE::convert_lrn() {
    auto input_0 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto input_1 = std::make_shared<pattern::op::Label>(element::i64, Shape{1});
    auto lrn = std::make_shared<ngraph::opset1::LRN>(input_0, input_1, 1, 1, 1, 1);

    ngraph::graph_rewrite_callback callback = [](pattern::Matcher& m) {
        auto lrn = std::dynamic_pointer_cast<ngraph::opset1::LRN> (m.get_match_root());
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
        ngraph::replace_node(m.get_match_root(), lrn_ie);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(lrn, "ConvertLRNToLRNIE");
    this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
}