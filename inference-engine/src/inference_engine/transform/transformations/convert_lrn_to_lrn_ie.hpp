// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>
#include <string>

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph_ops/lrn_ie.hpp>

#include "ngraph/op/lrn.hpp"
#include "ngraph/op/constant.hpp"

namespace ngraph {
namespace pass {

class ConvertLRNToLRNIE;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertLRNToLRNIE: public ngraph::pass::GraphRewrite {
public:
    ConvertLRNToLRNIE() : GraphRewrite() {
        convert_lrn();
    }

private:
    void convert_lrn();
};

void ngraph::pass::ConvertLRNToLRNIE::convert_lrn() {
    auto input_0 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto input_1 = std::make_shared<pattern::op::Label>(element::i64, Shape{1});
    auto proposal = std::make_shared<ngraph::op::LRN>(input_0, input_1, 1, 1, 1, 1);

    ngraph::graph_rewrite_callback callback = [](pattern::Matcher& m) {
        auto lrn = std::dynamic_pointer_cast<ngraph::op::LRN> (m.get_match_root());
        if (!lrn) {
            return false;
        }

        auto axis = std::dynamic_pointer_cast<ngraph::op::Constant> (lrn->input(1).get_source_output().get_node_shared_ptr());
        if (!axis) {
            return false;
        }

        auto const_shape = axis->get_output_shape(0);
        std::string region;
        if (const_shape.size() == 1 && const_shape[0] == 1) {
            region = "across";
        } else {
            return false;
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

    auto m = std::make_shared<ngraph::pattern::Matcher>(proposal, "CPUFusion.ConvertLRNToLRNIE");
    this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
}
