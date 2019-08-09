// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph_ops/prior_box_ie.hpp>

#include "ngraph/op/experimental/dyn_slice.hpp"
#include "ngraph/op/experimental/layers/prior_box.hpp"

namespace ngraph {
namespace pass {

class ConvertPriorBox;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertPriorBox: public ngraph::pass::GraphRewrite {
public:
    ConvertPriorBox() : GraphRewrite() {
        convert_prior_box();
    }

private:
    void convert_prior_box();
};

void ngraph::pass::ConvertPriorBox::convert_prior_box() {
    auto data = std::make_shared<pattern::op::Label>(element::i64, Shape{1, 1, 1, 1});
    auto image = std::make_shared<pattern::op::Label>(element::i64, Shape{1, 1, 1, 1});

    ngraph::op::PriorBoxAttrs attr;
    attr.min_size = {162.0f};
    attr.max_size = {213.0f};
    attr.aspect_ratio = {2.0f, 3.0f};
    attr.variance = {0.1f, 0.1f, 0.2f, 0.2f};
    attr.step = 64.0f;
    attr.offset = 0.5f;
    attr.clip = 0;
    attr.flip = 1;
    attr.scale_all_sizes = true;

    auto prior_box = std::make_shared<ngraph::op::PriorBox>(data, image, attr);

    ngraph::graph_rewrite_callback callback = [](pattern::Matcher& m) {
        auto prior_box_node = std::dynamic_pointer_cast<ngraph::op::PriorBox> (m.get_match_root());
        auto strided_slice1 = std::dynamic_pointer_cast<ngraph::op::DynSlice> (prior_box_node->get_argument(0));
        auto strided_slice2 = std::dynamic_pointer_cast<ngraph::op::DynSlice> (prior_box_node->get_argument(1));

        if (!strided_slice1 || !strided_slice2 || !prior_box_node) {
            return false;
        }

        //  Check that StridedSlice1 cuts H,W dims for PriorBox
        auto begin = std::dynamic_pointer_cast<ngraph::op::Constant> (strided_slice1->get_argument(1));
        auto end = std::dynamic_pointer_cast<ngraph::op::Constant> (strided_slice1->get_argument(2));
        auto stride = std::dynamic_pointer_cast<ngraph::op::Constant> (strided_slice1->get_argument(3));

        if (!begin || !end || !stride) {
            return false;
        }

        auto begin_val = begin->get_vector<int64_t>();
        auto end_val = end->get_vector<int64_t>();
        auto stride_val = stride->get_vector<int64_t>();

        if (begin_val.size() != 1 && begin_val[0] != 2) {
            return false;
        }

        if (end_val.size() != 1 && end_val[0] != 4) {
            return false;
        }

        if (stride_val.size() != 1 && stride_val[0] != 1) {
            return false;
        }

        // TODO: should we check second StridedSlice?

        auto shape_of1 = std::dynamic_pointer_cast<ngraph::op::ShapeOf> (strided_slice1->get_argument(0));
        auto shape_of2 = std::dynamic_pointer_cast<ngraph::op::ShapeOf> (strided_slice2->get_argument(0));

        if (!shape_of1 || !shape_of2) {
            return false;
        }

        auto prior_box_ie = std::make_shared<ngraph::op::PriorBoxIE> (shape_of1->get_argument(0),
                                                                      shape_of2->get_argument(0),
                                                                      prior_box_node->get_attrs());
        prior_box_ie->set_friendly_name(prior_box_node->get_friendly_name());
        ngraph::replace_node(m.get_match_root(), prior_box_ie);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(prior_box, "CPUFusion.ConvertPriorBoxToPriorBoxIE");
    this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
}
