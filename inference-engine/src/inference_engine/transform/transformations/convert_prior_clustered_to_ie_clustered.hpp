// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph_ops/prior_box_clustered_ie.hpp>

#include "ngraph/op/strided_slice.hpp"
#include "ngraph/op/experimental/layers/prior_box_clustered.hpp"

namespace ngraph {
namespace pass {

class ConvertPriorBoxClustered;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertPriorBoxClustered: public ngraph::pass::GraphRewrite {
public:
    ConvertPriorBoxClustered() : GraphRewrite() {
        convert_prior_box_clustered();
    }

private:
    void convert_prior_box_clustered() {
        auto data = std::make_shared<pattern::op::Label>(element::i64, Shape{1, 1, 1, 1});
        auto axes = ngraph::op::Constant::create(element::i64, Shape{1}, {0});
        auto image = std::make_shared<pattern::op::Label>(element::i64, Shape{1, 1, 1, 1});

        ngraph::op::PriorBoxClusteredAttrs attr;
        attr.widths = {0.1f, 0.1f, 0.2f, 0.2f};
        attr.heights = {0.1f, 0.1f, 0.2f, 0.2f};
        attr.variances = {0.1f, 0.1f, 0.2f, 0.2f};
        attr.step_widths = 64.0f;
        attr.step_heights = 64.0f;
        attr.offset = 0.5f;
        attr.clip = false;

        auto prior_box = std::make_shared<ngraph::op::PriorBoxClustered>(data, image, attr);
        auto unsqueeze = std::make_shared<ngraph::op::Unsqueeze> (prior_box, axes);

        ngraph::graph_rewrite_callback callback = [](pattern::Matcher& m) {
            auto unsqueeze = std::dynamic_pointer_cast<ngraph::op::Unsqueeze> (m.get_match_root());
            if (!unsqueeze) {
                return false;
            }
            auto prior_box_node = std::dynamic_pointer_cast<ngraph::op::PriorBoxClustered> (unsqueeze->get_argument(0));

            if (!prior_box_node) {
                return false;
            }

            std::shared_ptr<Node> input_1(prior_box_node->input_value(0).get_node_shared_ptr());
            std::shared_ptr<Node> input_2(prior_box_node->input_value(1).get_node_shared_ptr());

            auto convert1 = std::dynamic_pointer_cast<ngraph::op::Convert> (input_1);
            auto convert2 = std::dynamic_pointer_cast<ngraph::op::Convert> (input_2);

            if (convert1 && convert2) {
                input_1 = convert1->input_value(0).get_node_shared_ptr();
                input_2 = convert2->input_value(0).get_node_shared_ptr();
            }

            auto strided_slice1 = std::dynamic_pointer_cast<ngraph::op::v1::StridedSlice> (input_1);
            auto strided_slice2 = std::dynamic_pointer_cast<ngraph::op::v1::StridedSlice> (input_2);

            if (!strided_slice1 || !strided_slice2) {
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
            input_1 = strided_slice1->input_value(0).get_node_shared_ptr();
            input_2 = strided_slice2->input_value(0).get_node_shared_ptr();

            convert1 = std::dynamic_pointer_cast<ngraph::op::Convert> (input_1);
            convert2 = std::dynamic_pointer_cast<ngraph::op::Convert> (input_2);

            if (convert1 && convert2) {
                input_1 = convert1->input_value(0).get_node_shared_ptr();
                input_2 = convert2->input_value(0).get_node_shared_ptr();
            }

            auto shape_of1 = std::dynamic_pointer_cast<ngraph::op::ShapeOf> (input_1);
            auto shape_of2 = std::dynamic_pointer_cast<ngraph::op::ShapeOf> (input_2);

            if (!shape_of1 || !shape_of2) {
                return false;
            }

            auto prior_box_ie = std::make_shared<ngraph::op::PriorBoxClusteredIE> (shape_of1->get_argument(0),
                    shape_of2->get_argument(0),
                    prior_box_node->get_attrs());
            prior_box_ie->set_friendly_name(unsqueeze->get_friendly_name());
            ngraph::replace_node(m.get_match_root(), prior_box_ie);
            return true;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(unsqueeze, "CPUFusion.ConvertPriorBoxClusteredToPriorBoxClusteredIE");
        this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
    }
};
